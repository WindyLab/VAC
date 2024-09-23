from gvae.utils.flags import build_flags
import gvae.models.model_builder as model_builder
from gvae.datasets.small_synth_data import SmallSynthData
from gvae.datasets.small_synth_data import *

import gvae.training.evaluate as evaluate
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
from common_utils import eval_edges,eval_traj_prediction
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


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
    print("batch size:",args.batch_size)
    params['num_vars'] = 5
    params['input_size'] = 4
    params['input_time_steps'] = 24
    params['nll_loss_type'] = 'gaussian'
    params['num_edge_types'] = 2

    misc.seed(args.seed)
    train_loader, val_loader, test_loader,N_agent,N_time,N_data,N_feature_dim = prepare_flocking_dataset(args)

    model = model_builder.build_model(params)
    print(args.working_dir)
    forward_pred = params['input_time_steps'] - args.test_burn_in_steps
    test_mse  = eval_traj_prediction(model, test_loader, args.test_burn_in_steps, forward_pred, params)
    # path = os.path.join(args.working_dir, args.error_out_name%args.test_burn_in_steps)
    # np.save(path, test_mse.cpu().numpy())

    print("FORWARD PRED RESULTS:",test_mse)
    accuracy_,precision_,recall_,f1_,acc_std,pre_std,rec_std,f1_std, all_edges = eval_edges(model, test_loader, params)
    print("Val Edge results:")
    print("\tF1: ",f1_)
    print("\tAll predicted edge accuracy: ",accuracy_)
    print("\tAll predicted edge precision: ",precision_)
    print("\tAll predicted edge recall: ",recall_)
    print("\tacc_std: ",acc_std)
    print("\tpre_std: ",pre_std)
    print("\trec_std: ",rec_std)
    print("\tf1_std: ",f1_std)

