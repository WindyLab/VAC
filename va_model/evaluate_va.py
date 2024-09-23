import torch
import numpy as np
import os
import logging
from torch.utils.data import DataLoader,random_split
from models.va_net import VattNet
from cfg_vatt import config_train as cfg
from models.model_utils import evaluate
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import pdb
from time import time
from data_set import *

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

    begin_time = time()
    
    ## 1. build dataset and dataloader
    data_id = cfg['data_id']
    if cfg['with_motion']:
        img_dir = f'gazebo_motion_data_{data_id}/data'
        vattention_label_path = f'gazebo_motion_data_{data_id}/att_labels/'
        n_channel = 2
    else:
        img_dir = f'gazebo_data_{data_id}/data'
        vattention_label_path = f'gazebo_data_{data_id}/att_labels/'
        n_channel = 3
    checkpoint_dir = cfg['checkpoint_dir']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_set = VSwarmDataset(img_dir,vattention_label_path,scale=0.5,motion=cfg['with_motion'])

    ## 2. Split into train / validation partitions
    val_percent = cfg['val_percent']
    n_val = int(len(data_set) * val_percent)
    n_train = len(data_set) - n_val
    
    train_set, val_set = random_split(data_set, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=os.cpu_count(), pin_memory=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=cfg['batch_size'],
                                                    shuffle=True,num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, drop_last=True, **loader_args)

    # for batch_idx, data in enumerate(train_data_loader):
    #     inputs, target = data[0].to(device), data[1].to(device)
    #     print(batch_idx)
    # exit(0)

    ## 3. Build model and load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VattNet(n_channels=n_channel, n_classes=1, bilinear=True)
    weights = torch.load(cfg['model_dir'], map_location=lambda storage, loc: storage)
    model.load_state_dict(weights)
    model.to(device=device)

    ## 4. Start to evaluate
    KL_criterion = torch.nn.KLDivLoss(size_average=False,reduce="batchmean",log_target=True)
    loss,roc_auc,corr = evaluate(model, val_loader, device, KL_criterion,show_curve=False)
    print(f'corr:{corr}')
    print(f'roc_auc:{roc_auc}')