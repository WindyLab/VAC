import torch
import numpy as np
import os
import logging
from torch.utils.data import DataLoader,random_split
from models.va_net import VattNet
from cfg_vatt import config_train as cfg
from torch import optim
import torch.nn as nn
from models.model_utils import HeatmapLoss,evaluate
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import pdb
from torch.utils.tensorboard import SummaryWriter
from time import time
import path_utils
import thop
from copy import deepcopy
from data_set import *

def model_info(model, verbose=False, imgsz=640):
    """
    Prints model summary including layers, parameters, gradients, and FLOPs; imgsz may be int or list.

    Example: img_size=640 or img_size=[640, 320]
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1e9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f", {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs"  # 640x640 GFLOPs
    except Exception:
        fs = ""
    print(fs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    begin_time = time()

    ## 1. build dataset and dataloader
    data_id = cfg['data_id']
    for data_id in range(1):
        # if cfg['with_motion']:
        #     img_dir = f'gazebo_motion_data_{data_id}/data'
        #     vattention_label_path = f'gazebo_motion_data_{data_id}/att_labels/'
        #     n_channel = 2
        # else:
        #     img_dir = f'gazebo_data_{data_id}/data'
        #     vattention_label_path = f'gazebo_data_{data_id}/att_labels/'
        #     n_channel = 3

        if cfg['with_motion']:
            img_dir = f'real_world_motion_data_{data_id}/data'
            vattention_label_path = f'real_world_motion_data_{data_id}/att_labels/'
            n_channel = 2
        else:
            img_dir = f'real_world_data_{data_id}/data'
            vattention_label_path = f'real_world_data_{data_id}/att_labels/'
            n_channel = 1

        checkpoint_dir = cfg['checkpoint_dir']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        path_folder = path_utils.create_path_to_folder(checkpoint_dir)
        
        if data_id == 0:
            data_set = VSwarmDataset(img_dir,vattention_label_path,scale=0.25,ch = n_channel,motion=cfg['with_motion'])
        else:
            data_set += VSwarmDataset(img_dir,vattention_label_path,scale=0.25,ch = n_channel, motion=cfg['with_motion'])
    
    # print(len(data_set))
    # exit()
    
    n_channel = 2 if cfg['with_motion'] else 1
    
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
    #     #pdb.set_trace()
    # #exit(0)

    ## 3. Build network model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ##channel = [8,16,32,64,128]
    ## [16,32,64,128,256]
    ## [32,64,128,256,512]
    channel = cfg['va_channel']
    bilinear = cfg['bilinear']
    model = VattNet(n_channels=n_channel, n_classes=1, channel = channel,bilinear=bilinear)
    if cfg['load_pretrained'] == True:
        weights = torch.load(cfg['pretrained_path'], map_location=lambda storage, loc: storage)
        model.load_state_dict(weights)
    model.to(device=device)
    
    ## Print model train
    # model_info(model,True)

    ## 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=cfg['learning_rate'], 
                              weight_decay=cfg['weight_decay'],
                              foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    criterion = HeatmapLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
    
    # ## 5. Begin training
    epochs = cfg['epochs']
    global_step = 0
    KL_criterion = torch.nn.KLDivLoss(size_average=False,reduce="mean",log_target=True)
    iters_per_epoch = len(train_data_loader)
    WARM_UP_EPOCH = 3
    warm_up_epoch = WARM_UP_EPOCH
    warm_up_iters = warm_up_epoch * iters_per_epoch
    train_iters = (epochs - warm_up_epoch) * iters_per_epoch
    initial_lr = cfg['learning_rate']
    smm_folder = path_utils.create_path_to_folder('./run_log/')
    writer = SummaryWriter(smm_folder)
    try:
        for epoch in range(0, epochs + 1):
            model.train()
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch_idx, data in enumerate(train_data_loader):
                    inputs, target = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()
                    # # forward + backward + optimize
                    predicted = model(inputs)
                   # print("predicted shape:",predicted.shape)
                   # print("target shape:",target.shape)
                   # pdb.set_trace()
                    predicted = torch.log_softmax(predicted,dim=-1)
                    target = torch.log_softmax(target,dim=-1)

                    loss = KL_criterion(predicted,target)
                   # loss = torch.mean(loss)  #equivalent to reduction = mean, see offical description for loss
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    pbar.update(inputs.shape[0])
                    optimizer.step()

                    # print statistics
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    global_step += 1
                    # adjust learning rate
                    current_iters = epoch * iters_per_epoch + batch_idx
                    # if current_iters < warm_up_iters:
                    #     lr = initial_lr * 0.1 + current_iters / warm_up_iters * initial_lr * 0.9
                    # else:
                    #     lr = (1 - (current_iters - warm_up_iters) / train_iters) * initial_lr
                        
                    # for param in optimizer.param_groups:
                    #     param['lr'] = lr

                if epoch % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.15f}'
                     .format(epoch+1, epochs, batch_idx+1, len(train_data_loader), loss.item()))
                    print("====learning rate,",optimizer.state_dict()['param_groups'][0]['lr'])
                if epoch % cfg['save_epoch'] == 0:
                    path_model = path_utils.path_to_model(path_folder,epoch=epoch)
                    print(path_model)
                    state_dict = model.state_dict()
                    torch.save(state_dict, path_model)
                    logging.info(f'Checkpoint {epoch} saved!')
            scheduler.step()
            
            # Evaluation round
            if epoch % 5 == 0:
                val_score,roc_auc,corr = evaluate(model, val_loader, device, KL_criterion)
                writer.add_scalar('Loss/val',scalar_value = val_score, global_step=epoch)
                writer.add_scalar('Loss/train',scalar_value = loss.item(), global_step=epoch)
                writer.add_scalar('roc_auc',scalar_value = roc_auc, global_step=epoch)
                writer.add_scalar('corr',scalar_value = corr, global_step=epoch)
                logging.info(f'evaluation {epoch} ok! loss:{val_score}')

    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
    writer.close()
    duation = time() - begin_time
    print("Initial training duation:",duation)