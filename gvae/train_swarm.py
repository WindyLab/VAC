from gvae.utils.flags import build_flags
import gvae.models.model_builder as model_builder
from gvae.datasets.small_synth_data import SmallSynthData
from gvae.datasets.small_synth_data import *

import gvae.training.train_utils as train_utils
import gvae.utils.misc as misc

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
import pdb
import time
from common_utils import eval_edges
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train_swarm(model, train_data_loader, val_data_loader, params, train_writer, val_writer):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    val_batch_size = params.get('val_batch_size', batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    accumulate_steps = params.get('accumulate_steps')
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 1)
    val_start = params.get('val_start', 0)
    clip_grad = params.get('clip_grad', None)
    clip_grad_norm = params.get('clip_grad_norm', None)
    normalize_nll = params.get('normalize_nll', False)
    normalize_kl = params.get('normalize_kl', False)
    tune_on_nll = params.get('tune_on_nll', False)
    verbose = params.get('verbose', False)
    val_teacher_forcing = params.get('val_teacher_forcing', False)
    continue_training = params.get('continue_training', False)

    lr = params['lr']
    wd = params.get('wd', 0.)
    mom = params.get('mom', 0.)
    
    model_params = [param for param in model.parameters() if param.requires_grad]
    if params.get('use_adam', False):
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom)

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    if continue_training:
        print("RESUMING TRAINING")
        model.load(checkpoint_dir)
        train_params = torch.load(training_path)
        start_epoch = train_params['epoch']
        opt.load_state_dict(train_params['optimizer'])
        best_val_result = train_params['best_val_result']
        best_val_epoch = train_params['best_val_epoch']
        print("STARTING EPOCH: ",start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_result = 10000000
    
    training_scheduler = train_utils.build_scheduler(opt, params)
    end = start = 0 
    misc.seed(1)
    
    # for i, data in enumerate(train_data_loader):
    #     print(data[0].shape,data[1].shape)
    #     print(data[0][0][0][0],data[1][0][0][0])
   # print(len(train_data_loader))
   # pdb.set_trace()
   
    for name, p in model.named_parameters():
        if name == 'kl_coef.0':
            p.data.clamp_(0, 1)
            
            
    for epoch in range(start_epoch, num_epochs+1):
        print("EPOCH", epoch, (end-start))
        model.train()
        model.train_percent = epoch / num_epochs
        start = time.time()
        for batch_ind, batch in enumerate(train_data_loader):
            inputs = batch[0]   #batch['inputs']
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
            loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(inputs, is_train=True, return_logits=True)

            loss.backward()
            if verbose:
                print("\tBATCH %d OF %d: %f, %f, %f"%(batch_ind+1, len(train_data_loader), loss.item(), loss_nll.mean().item(), loss_kl.mean().item()))
            if accumulate_steps == -1 or (batch_ind+1)%accumulate_steps == 0:
                if verbose and accumulate_steps > 0:
                    print("\tUPDATING WEIGHTS")
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(model.parameters(), clip_grad)
                elif clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                opt.step()
                opt.zero_grad()
                if accumulate_steps > 0 and accumulate_steps > len(train_data_loader) - batch_ind - 1:
                    break
            for name, p in model.named_parameters():
                if name == 'kl_coef.0':
                    p.data.clamp_(1e-09, 1)
            
        if training_scheduler is not None:
            training_scheduler.step()
            print(training_scheduler.get_lr())

        if train_writer is not None:
            train_writer.add_scalar('loss', loss.item(), global_step=epoch)
            if normalize_nll:
                train_writer.add_scalar('NLL', loss_nll.mean().item(), global_step=epoch)
            else:
                train_writer.add_scalar('NLL', loss_nll.mean().item()/(inputs.size(1)*inputs.size(2)), global_step=epoch)
            
            train_writer.add_scalar("KL Divergence", loss_kl.mean().item(), global_step=epoch)
        model.eval()
        opt.zero_grad()

        total_nll = 0
        total_kl = 0
        if verbose:
            print("COMPUTING VAL LOSSES")
        with torch.no_grad():
            ###################
            accuracy_,precision_,recall_,f1_,acc_std,pre_std,rec_std,f1_std, all_edges = eval_edges(model, val_data_loader, params)
            print("Val Edge results:")
            print("\tF1: ",f1_)
            print("\tAll predicted edge accuracy: ",accuracy_)
            print("\tAll predicted edge precision: ",precision_)
            print("\tAll predicted edge recall: ",recall_)
            
            print("\tacc_std: ",acc_std)
            print("\tpre_std: ",pre_std)
            print("\trec_std: ",rec_std)
            print("\tf1_std: ",f1_std)
            
            for batch_ind, batch in enumerate(val_data_loader):
                inputs = batch[0]
                if gpu:
                    inputs = inputs.cuda(non_blocking=True)
                loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(inputs, is_train=False, teacher_forcing=val_teacher_forcing, return_logits=True)
                total_kl += loss_kl.sum().item()
                total_nll += loss_nll.sum().item()
                if verbose:
                    print("\tVAL BATCH %d of %d: %f, %f"%(batch_ind+1, len(val_data_loader), loss_nll.mean(), loss_kl.mean()))
            
        total_kl /= len(val_data_loader.dataset)
        total_nll /= len(val_data_loader.dataset)
        total_loss = model.kl_coef*total_kl + total_nll #TODO: this is a thing you fixed
        
       # pdb.set_trace()
        if val_writer is not None:
            val_writer.add_scalar('loss', total_loss, global_step=epoch)
            val_writer.add_scalar("NLL", total_nll, global_step=epoch)
            val_writer.add_scalar("KL Divergence", total_kl, global_step=epoch)
            val_writer.add_scalar("F1",f1_,global_step=epoch)
            val_writer.add_scalar("Acc",accuracy_,global_step=epoch)

        if tune_on_nll:
            tuning_loss = total_nll
        else:
            tuning_loss = total_loss
        if tuning_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = tuning_loss
            print("BEST VAL RESULT. SAVING MODEL...")
            model.save(best_path)
        model.save(checkpoint_dir)
        torch.save({
                    'epoch':epoch+1,
                    'optimizer':opt.state_dict(),
                    'best_val_result':best_val_result,
                    'best_val_epoch':best_val_epoch,
                   }, training_path)
        print("EPOCH %d EVAL: "%epoch)
        print("\tCURRENT VAL LOSS: %f"%tuning_loss)
        print("\tBEST VAL LOSS:    %f"%best_val_result)
        print("\tBEST VAL EPOCH:   %d"%best_val_epoch)
        end = time.time()

if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--same_data_norm', action='store_true')
    parser.add_argument('--no_data_norm', action='store_true')
    parser.add_argument('--error_out_name', default='prediction_errors_%dstep.npy')
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--test_burn_in_steps', type=int, default=10)
    parser.add_argument('--error_suffix')
    parser.add_argument('--subject_ind', type=int, default=-1)
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--obs_frames', type=int, default=24)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data/swarm/data_flocking_55_50000_24_10.npy')
    parser.add_argument('--label_path', type=str, default='data/swarm/label_flocking_55_50000_24_10.npy')

    args = parser.parse_args()
    params = vars(args)
    print("batch size:",args.batch_size,args.data_path)
    params['nll_loss_type'] = 'gaussian'
    params['num_edge_types'] = 2

    misc.seed(args.seed)
    train_loader, val_loader, test_loader, N_agent,N_time,N_data,N_feature_dim = prepare_flocking_dataset(args)
    params['num_vars'] = N_agent
    params['input_size'] = N_feature_dim
    params['input_time_steps'] = N_time
    print(f'N_agent {N_agent} N_feature_dim {N_feature_dim} N_time {N_time}')

    model = model_builder.build_model(params)
    with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
        train_swarm(model, train_loader, val_loader, params, train_writer, val_writer)
    train_writer.close()