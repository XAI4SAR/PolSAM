import argparse
import math
import os
import random
import sys
import time
import wandb

import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
import torchvision.utils as vutils
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import MydatasetP, Test, MydatasetSF_Ablation
from utils import (FocalLoss, acc_etc, lr_schedule_cosine)


def trainer_SAR(cfg, model, experiment_path, multimask_output, data_info, device):
    Classes, colormap, class_names = data_info['classes'], data_info['colormap'], data_info['class_names']
    base_lr = cfg.base_lr
    batch_size = cfg.batch_size * cfg.n_gpu
    save_path = os.path.join(experiment_path, 'evaluation.txt')
    save_model = os.path.join(experiment_path, 'result.pth')
    def worker_init_fn(worker_id):
        random.seed(cfg.seed + worker_id)
    if cfg.dataset_name == 'coarse' or cfg.dataset_name == 'SF':
        my_datat = MydatasetP(inputnum=2, data_info=data_info, cfg=cfg, bs=4, train=True, aug=True, onehot=True)
        my_datav = MydatasetP(inputnum=2, data_info=data_info, cfg=cfg, bs=4, train=False, aug=True, onehot=True)
        # my_datat = MydatasetSF_Ablation(data_info=data_info, cfg=cfg, train=True, aug=False)  # Ablation dataset class
        # my_datav = MydatasetSF_Ablation(data_info=data_info, cfg=cfg, train=False, aug=False)
    if cfg.n_gpu > 1:
        sampler = DistributedSampler(dataset)
        trainloader = DataLoader(my_datat, batch_size=batch_size, sampler=sampler, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
        valloader = DataLoader(my_datav, batch_size=batch_size, sampler=sampler, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    else:
        trainloader = DataLoader(my_datat, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
        valloader = DataLoader(my_datav, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    max_epoch = cfg.max_epochs
    max_iterations = cfg.max_epochs * len(trainloader)
    warmup_period = max_iterations * cfg.warmup_rate
    print("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))
    if cfg.warmup:
        b_lr = base_lr / warmup_period
    else:
        b_lr = base_lr
    if cfg.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    best_score = 0.0
    start_time = time.time()
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        print(f'------第{epoch+1}轮训练开始------')
        model.train()
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        for pack in trainloader:
            image, mx, mxonehot, label = pack['image'].cuda(device), pack['scatrgb'].cuda(device), pack['scatonehot'].cuda(device), pack['label'].cuda(device)
            output = model(image, multimask_output, cfg.img_size, mx, mxonehot, mx)
            
            if data_info['weight'] is not None:
                weights = torch.tensor(data_info['weight']).cuda(device)
            if cfg.loss == 'ce':
                loss_fun = nn.CrossEntropyLoss()
                loss  = loss_fun(output, label)
            elif cfg.loss == 'wce':
                loss_function = CrossEntropyLoss(weight=weights)
                loss = loss_function(output, label)
            elif cfg.loss == 'focal':
                lossfocal = FocalLoss(alpha=weights, num_classes=Classes)
                loss = lossfocal(output, label)
            elif cfg.loss == 'dice':
                loss_fun = DiceLoss(n_classes=Classes)
                loss = loss_fun(output, label, weight=weights, softmax=True)
            elif cfg.loss == 'ohem':
                loss_fun = OhemLoss(ignore=255)
                loss = loss_fun(output, label)

            pred = output.argmax(dim=1).squeeze().data.cpu()
            real = label.data.cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cfg.warmup and iter_num < warmup_period:
                lr_ = base_lr * ((iter_num + 1) / warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if cfg.warmup:
                    shift_iter = iter_num - warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                    
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            train_loss += loss.cpu().item()*label.size(0)
            label_true = torch.cat((label_true, real), dim=0)
            label_pred = torch.cat((label_pred, pred), dim=0)


        train_loss /= len(my_datat)
        acc, train_iou, pre, M_Recall, F1_score, MIOU_no_back, MIOU = acc_etc(label_true.numpy(), label_pred.numpy(), n_class=Classes)
        print('train_loss:{:.4f}, lr:{:.6f}, acc:{:.4f}, pre:{:.4f}, recall:{:.4f}, F1_score:{:.4f}, MIOU_no_back:{:.4f}, MIOU:{:.4f}, epoch:{}'.format(train_loss, lr_, acc, pre, M_Recall, F1_score, MIOU_no_back, MIOU,epoch+1))
        with open(save_path, 'a') as f:
            f.write('\n epoch:{}, train_loss:{:.4f}, lr:{:.6f}, acc:{:.4f}, F1_score:{:.4f}, MIOU:{:.4f}'.format(
                epoch+1, train_loss, lr_, acc, F1_score, MIOU))
        model.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        with torch.no_grad():
            for pack in valloader:
                image, mx, mxonehot, label = pack['image'].cuda(device), pack['scatrgb'].cuda(device), pack['scatonehot'].cuda(device), pack['label'].cuda(device)
                output = model(image, multimask_output, cfg.img_size, mx, mxonehot, mx)

                if data_info['weight'] is not None:
                    weights = torch.tensor(data_info['weight']).cuda(device)
                if cfg.loss == 'ce':
                    loss_fun = nn.CrossEntropyLoss()
                    loss  = loss_fun(output, label)
                elif cfg.loss == 'wce':
                    loss_function = CrossEntropyLoss(weight=weights)
                    loss = loss_function(output, label)
                elif cfg.loss == 'focal':
                    lossfocal = FocalLoss(alpha=weights, num_classes=Classes)
                    loss = lossfocal(output, label)
                elif cfg.loss == 'dice':
                    loss_fun = DiceLoss(n_classes=Classes)
                    loss = loss_fun(output, label, weight=weights)
                elif cfg.loss == 'ohem':
                    loss_fun = OhemLoss(ignore=255)
                    loss = loss_fun(output, label)

                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = label.data.cpu()
                
                val_loss += loss.cpu().item()*label.size(0)
                val_label_true = torch.cat((val_label_true, real), dim=0)
                val_label_pred = torch.cat((val_label_pred, pred), dim=0)
                
            val_loss /= len(my_datav)
            val_acc, val_iou, val_pre, val_M_Recall, val_F1_score, val_MIOU_no_back, val_MIOU = acc_etc(val_label_true.numpy(), val_label_pred.numpy(), n_class=Classes)
        print('val_loss:{:.4f}, acc:{:.4f}, pre:{:.4f}, recall:{:.4f}, F1_score:{:.4f}, MIOU_no_back:{:.4f}, MIOU:{:.4f}, epoch:{} '.format(
               val_loss, val_acc, val_pre, val_M_Recall, val_F1_score, val_MIOU_no_back, val_MIOU, epoch+1))

        with open(save_path, 'a') as f:
            f.write('\n epoch:{}, val_loss:{:.4f}, val_acc:{:.4f}, val_F1_score:{:.4f}, val_MIOU:{:.4f}'.format(
                epoch+1, val_loss, val_acc, val_F1_score, val_MIOU))
        score = (val_acc+val_F1_score+val_MIOU)/3
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_model)
            print("\033[33msave model to\033[0m {}".format(save_model))

        use_time = time.time() - start_time
        print('用时{:.0f}minutes {:.0f}s'.format(use_time // 60, use_time % 60))

    return "Training Finished!"
