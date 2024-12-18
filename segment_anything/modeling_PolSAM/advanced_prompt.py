import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

class FeatureFusionModule_SoAM(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath_S(dim=dim, reduction=reduction, num_heads=num_heads)
        # self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
        #                                 norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = segfeature.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2,x3)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x1,x2


class FeatureFusionModule_MoAM(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath_M(dim=dim, reduction=reduction, num_heads=num_heads)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = segfeature.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2,x3)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x1,x2


class Network3(nn.Module):
    def __init__(self,backbone, num_classes=20, embedding_dim=256, pretrained=True):
        super(Network3, self).__init__()
        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        # self.discriminator = Discriminator()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, fused_seg1):
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused_seg1,fused_seg1,seg_map
    def _loss(self,fused_seg1,label,criterion):
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = criterion(outputs,label.type(torch.long))
        return denoise_loss
    def enhance_net_parameters(self):
        return self.enhance_net.parameters()
    def denoise_net_parameters(self):
        return self.denoise_net.parameters()


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv3 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2, segfeature):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        # k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k3, v3 = self.kv3(segfeature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        # ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        # ctx1 = ctx1.softmax(dim=-2)
        # ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        # ctx2 = ctx2.softmax(dim=-2)

        ctx3 = (k3.transpose(-2, -1) @ v3) * self.scale
        ctx3 = ctx3.softmax(dim=-2)


        x1 = (q1 @ ctx3).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx3).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2

class CrossAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention2, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.kv3 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2, segfeature):
        B, N, C = x1.shape
        # q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q3 = segfeature.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # k3, v3 = self.kv3(segfeature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        #
        # ctx3 = (k3.transpose(-2, -1) @ v3) * self.scale
        # ctx3 = ctx3.softmax(dim=-2)


        x1 = (q3 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q3 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, classes=6, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj3 = nn.Linear(dim, dim // reduction * 2)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)

        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.cross_attn2 = CrossAttention2(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim // reduction * 4, classes)
        # self.end_proj1 = nn.Linear(dim // reduction * 4, 50 * classes)
        # self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)

        # self.norm1 = norm_layer(50 * classes)  # int(0.5 * classes)
        self.norm1 = norm_layer(classes)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2,segfeature):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        y3, u3 = self.act3(self.channel_proj3(segfeature)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2, u3)
        z1, z2 = self.cross_attn2(y1, y2, y3)
        y1 = (z1 + v1)/2 + x1  # b, 1024, 256
        y2 = torch.cat((z2, v2), dim=-1)  # b, 1024, 512
        out_x1 = self.norm1(self.end_proj1(y1.transpose(1, 2))).transpose(1, 2)
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class Fusion_Network3_ac(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ir = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_vis = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=64)
        self.DRDB2 = DRDB(in_ch=64)
        self.DRDB3 = DRDB(in_ch=64)
        self.DRDB4 = DRDB(in_ch=64)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        # self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.ffm = FeatureFusionModule(64)
        self.ffm2 = FeatureFusionModule(64)
        self.conv3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, 1, padding=0)
        self.conv21 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv22 = nn.Conv2d(32, 1, 3, padding=1)
    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')
        ir = ir[:, 0:1, :, :]
        x1 = self.conv1_ir(ir)
        x1 = self.relu(x1)
        x1 =self.DRDB1(x1)
        vis = vis[:, 0:1, :, :]
        x2 = self.conv1_vis(vis)
        x2 = self.relu(x2)
        x2 = self.DRDB2(x2)
        x1,x2 = self.ffm(x1,x2,self.conv3(out1))
        x1 = self.DRDB3(x1)
        x2 = self.DRDB4(x2)
        x1, x2 = self.ffm(x1, x2, self.conv4(out2))
        # f2 = self.skff2([f2,self.conv4(conv4)])
        f_final = self.relu(self.conv2((torch.cat([x1,x2],dim=1))))
        f_final= self.relu(self.conv21(f_final))
        f_final= self.relu(self.conv22(f_final))

        return f_final

class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, classes=6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads, classes=classes)
        # self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
        #                                 norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,segfeature):
        B, C, H, W = x1.shape  # b,256,32,32
        x1 = x1.flatten(2).transpose(1, 2)  # b,32*32,256
        x2 = x2.flatten(2).transpose(1, 2)
        x3 = segfeature.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2, x3)   # b,32*32,256
        # merge = torch.cat((x1, x2), dim=-1)
        # merge = self.channel_emb(merge, H, W)
        # x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x1, x2
