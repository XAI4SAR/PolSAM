import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from segment_anything import sam_model_registry
from trainer import trainer_SAR
from datasets.transforms import get_data_info
# from utils import get_data_info
from config import cfg
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


if __name__ == "__main__":
    if not cfg.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    cfg.output_dir =  cfg.output + cfg.dataset_name + '/' + cfg.experiment
    cfg.exp = cfg.dataset_name + '_' + str(cfg.img_size)
    experiment_path = os.path.join(cfg.output_dir, "{}".format(cfg.exp))
    experiment_path += '_' + cfg.vit_name
    experiment_path = experiment_path + '_epo' + str(cfg.max_epochs)
    experiment_path = experiment_path + '_bs' + str(cfg.batch_size)
    experiment_path = experiment_path + '_warm' + str(cfg.warmup_rate)
    experiment_path = experiment_path + '_loss' + str(cfg.loss)
    experiment_path = experiment_path + '_lr' + str(cfg.base_lr) if cfg.base_lr != 0.001 else experiment_path
    experiment_path = experiment_path + '_' + str(datetime.now().isoformat())[:16]  # .split('.')[-2]
    experiment_path = experiment_path + '_s' + str(cfg.seed) if cfg.seed != 1234 else experiment_path
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    data_info = get_data_info(cfg.dataset_name)
    Classes = data_info['classes']
    print(f"Dataset: {cfg.dataset_name}-------Classes: {Classes}")
    sam, img_embedding_size = sam_model_registry[cfg.vit_name](image_size=cfg.img_size,
                                                                num_classes=Classes,
                                                                pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1],
                                                                checkpoint=cfg.ckpt,
                                                                pt=cfg.pt,
                                                                adapter=cfg.adapter,
                                                                Cnn_extra=cfg.Cnn_extra,
                                                                )
    if cfg.n_gpu > 1:
        print(".............distributed training.............")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(cfg.n_gpu)
        os.environ['MASTER_ADDR'] = cfg.master_addr
        os.environ['MASTER_PORT'] = '12355'

        torch.distributed.init_process_group(backend='nccl', init_method='env://',)
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        net = sam.to(device)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        device = torch.device('cuda', cfg.device)
        net = sam.to(device)

    if Classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(experiment_path, 'config.txt')
    config_items = []
    for key, value in cfg.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer_SAR(cfg, net, experiment_path, multimask_output, data_info, device)

# python train.py
