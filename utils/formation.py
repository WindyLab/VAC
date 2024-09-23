import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
from common_utils import *
import torch
from formation_utils import *
from matplotlib.patches import Rectangle
import argparse
from tqdm import tqdm
import os
import shutil
from trajectory_dynamic import *
from trajectory_static import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset_size', type=int, default=8000)
    parser.add_argument('--randomseed', type=int, default=199)
    parser.add_argument('--obs_frames', type=int, default=24)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--num_main', type=int, default=4)
    parser.add_argument('--num_unre', type=int, default=2)
    parser.add_argument('--feat_dim', type=int, default=2)
    
    parser.add_argument('--motion_mode', type=int, default=1)  ## 0 : retan 1: circular
    parser.add_argument('--perception_radius', type=float, default=2)
    parser.add_argument('--intershow', type=bool, default=False)
    parser.add_argument('--save_ani', type=bool, default=False)
    parser.add_argument('--collision_warning', type=bool, default=False)
    
    args = parser.parse_args()
    obs_frames = args.obs_frames
    rollouts = args.rollouts
    seq_length = rollouts + obs_frames
    feat_dim = 2
    torch.manual_seed(args.randomseed)
    num_instances = args.dataset_size
    target_path = './'
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # else:
    #     shutil.rmtree(target_path,ignore_errors=True)
    #     os.makedirs(target_path)

    data_path = f'{target_path}/data_flocking_{args.randomseed}_{args.dataset_size}_{args.obs_frames}_{args.rollouts}_{args.num_main + args.num_unre}.npy'
    label_path = f'{target_path}/label_flocking_{args.randomseed}_{args.dataset_size}_{args.obs_frames}_{args.rollouts}_{args.num_main + args.num_unre}.npy'

    all_data = []
    all_labels = []
    for test_case in tqdm(range(num_instances)):
        data,label = trajec_test(args)
        all_data.append(data)
        all_labels.append(label)
    all_data = np.stack(all_data)
    all_labels = np.stack(all_labels)
    
    print(f'all_data shape{all_data.shape}, all_labels shape{all_labels.shape}')
    np.save(data_path,all_data)
    np.save(label_path,all_labels)
    
    load_check_data = np.load(data_path)
    load_check_label = np.load(label_path)
    print(f'load_check_data shape{load_check_data.shape}, load_check_label shape{load_check_label.shape}')