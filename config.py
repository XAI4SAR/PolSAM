import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

cfg = edict()

# data
cfg.dataset_name = 'coarse' # coarse
# cfg.ablation = 'H_alpha'
cfg.img_size = 512
cfg.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
cfg.norm_mean = np.array([0.485, 0.456, 0.406])
cfg.norm_std = np.array([0.229, 0.224, 0.225])

# modal
# cfg.experiment = 'debug'
cfg.experiment = 'framework_Baseline'


cfg.pt = True
cfg.adapter = True
cfg.Cnn_extra = True

cfg.vit_name = 'vit_b'
# cfg.ckpt = './checkpoint/sam/sam_vit_l_0b3195.pth'
cfg.ckpt = './checkpoint/sam/sam_vit_b_01ec64.pth'
# train
cfg.output = './save_model/'
cfg.seed = 1234
cfg.max_epochs = 100
cfg.batch_size = 16
cfg.n_gpu = 1
cfg.device = 1
cfg.deterministic = 0
cfg.base_lr = 0.001

cfg.warmup = True
cfg.warmup_rate = 0.04  # The top 4% use warmup
cfg.AdamW = True
cfg.loss = 'focal'
# cfg.loss = 'ce'