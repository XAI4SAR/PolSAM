import argparse
import os
import time
from datetime import datetime
from importlib import import_module
from config import cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets.transforms import get_data_info
from dataset import as Test, MydatasetSF_Ablation
from segment_anything import sam_model_registry
from trainer import trainer_SAR
from utils import acc_etc, draw_ROC, output_up, show_results

parser = argparse.ArgumentParser()
parser.add_argument('--pt', type=bool, default=False)
parser.add_argument('--adapter', type=bool, default=False)
# parser.add_argument('--Cnn_extra', type=bool, default=False)
parser.add_argument('--Cnn_extra', type=bool, default=True)
parser.add_argument('--batch_size', type=int,default=16, help='batch_size per gpu')  # 12--vit-b, 8--vit--l
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--gpu_device', type=int, default=2, help='total gpu')
parser.add_argument('--img_size', type=int,default=512, help='input patch size of network input')
parser.add_argument('--vit_name', type=str,default='vit_b', help='select one vit model')
# parser.add_argument('--vit_name', type=str,default='vit_l', help='select one vit model')
parser.add_argument('--dataset_name', type=str, default='SF', help='dataset_name')
parser.add_argument('--ckpt', type=str, default='./checkpoint/sam/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--module', type=str, default='my_sam')
args = parser.parse_args()

if __name__ == "__main__":
    data_info = get_data_info(args.dataset_name)
    Classes = data_info['classes']
    GPUdevice = torch.device('cuda', args.gpu_device)
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=Classes,
                                                                pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1],
                                                                checkpoint=args.ckpt,
                                                                pt=args.pt,
                                                                adapter=args.adapter,
                                                                Cnn_extra=args.Cnn_extra,
                                                                )
    net = sam.to(device=GPUdevice)
    # netpath = r'save_model/coarse/framework_M6/coarse_512_vit_b_epo350_bs16_warm0.04_lossce_2024-09-13T10:43/result.pth'
    netpath = r'save_model/SF/framework_Baseline/SF_512_vit_b_epo100_bs16_warm0.04_lossfocal_2024-10-17T12:31/result.pth'
    save_path = os.path.dirname(netpath)
    net.load_state_dict(torch.load(netpath, map_location='cuda:2'), strict=True)
    net.eval()

    test_data = Test(args.dataset_name)
    # test_data = MydatasetSF_Ablation(data_info=data_info, cfg=cfg, train=False, aug=False)
    # my_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    my_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
    drawroc = False
    test_alloutput = torch.LongTensor()
    val_label_true = torch.LongTensor()
    val_label_pred = torch.LongTensor()
    start_time = time.time()
    for i, (img, scatrgb, scatonehot, pauliscat, img_label) in enumerate(my_dataloader):
        img, img_label, scatrgb, scatonehot = img.to(GPUdevice), img_label.to(GPUdevice), scatrgb.to(GPUdevice), scatonehot.to(GPUdevice)
    # for pack in my_dataloader:
    #     img, scatrgb, scatonehot, img_label = pack['image'].cuda(GPUdevice), pack['scatrgb'].cuda(GPUdevice), pack['scatonehot'].cuda(GPUdevice), pack['label'].cuda(GPUdevice)
        with torch.no_grad():
            output = net(img, True, args.img_size, img, img, img)
            # output = net(img, True, args.img_size, scatrgb, scatonehot, scatrgb)
            # output = net(img, True, args.img_size, scatrgb, scatonehot, scatonehot)
            output = F.log_softmax(output, dim=1)
            output = output_up(output, img_label)

            alloutput = output.data.cpu()
            pred = output.argmax(dim=1).squeeze().data.cpu()
            real = img_label.data.cpu()

            test_alloutput = torch.cat((test_alloutput, alloutput), dim=0)

            val_label_true = torch.cat((val_label_true, real), dim=0)
            val_label_pred = torch.cat((val_label_pred, pred), dim=0)
    use_time = time.time() - start_time
    print('测试用时{:.0f}minutes {:.0f}s'.format(use_time // 60, use_time % 60))
    if drawroc:
        roc = draw_ROC(test_alloutput, val_label_true, save_path)
    plt.clf()
    name_classes = data_info['class_names']
    acc, M_F1_score, MIOU, hist, iou, F1_score, Recall, Pre = acc_etc(val_label_true.numpy(), val_label_pred.numpy(), Classes, test=True)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    print(f'iou:{iou}, test_acc:{acc}, test_F1_score:{M_F1_score}, test_MIOU:{MIOU}')
    show_results(hist=hist, IoUs=iou, F1_Score=F1_score, Recall=Recall, Precision=Pre, miou_out_path=save_path, name_classes=name_classes)
    # print('----------------------------------------------------------------------------------')