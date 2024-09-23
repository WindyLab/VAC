import numpy as np
import torch
from torch.utils.data import Dataset
from gvae.utils import data_utils
import pickle
import argparse, os
import pdb

def load_pkl_data(mode,data_path,num_agent,len_sequence,num_sequence):
    loc_path = f'{data_path}/loc_{mode}_{num_agent}_{len_sequence}_{num_sequence}.pkl'
    vel_path = f'{data_path}/vel_{mode}_{num_agent}_{len_sequence}_{num_sequence}.pkl'
    edge_path = f'{data_path}/edge_{mode}_{num_agent}_{len_sequence}_{num_sequence}.pkl'
    
    with open(loc_path, 'rb') as f:
        loc_train  = pickle.load( f)
    with open(vel_path, 'rb') as f:
        vel_train = pickle.load( f)
    with open(edge_path, 'rb') as f:
        edge_train = pickle.load( f)
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    num_seq = loc_train.shape[0]
    num_agents = loc_train.shape[1]
    observe_frame = loc_train.shape[2]
 #  edges_train = np.reshape(edge_train, [-1,observe_frame,num_agents ** 2])
    print("edges_train shape:",edge_train.shape)
    print("feat_train shape:",loc_train.shape)
    print(f"number of agents {num_agents}, observe_frame {observe_frame},num_seq {num_seq}")
   # pdb.set_trace()
    return loc_train, vel_train, edge_train

class VicsekData(Dataset):
    def __init__(self,mode,data_path,num_agent,len_sequence,num_sequence):
        loc_train, vel_train, edges_train = load_pkl_data(mode,data_path,num_agent,len_sequence,num_sequence)
        super(VicsekData, self).__init__()
        self.num_agent = np.shape(loc_train)[1]
        self.num_frames = np.shape(loc_train)[2]
        self.num_sequences = np.shape(loc_train)[0]
        loc_train = np.transpose(loc_train,[0,2,1,3])
        vel_train = np.transpose(vel_train,[0,2,1,3])
        
        self.loc = torch.FloatTensor(loc_train)  ## data, seq_len, N,2
        self.vel = torch.FloatTensor(vel_train)  ## data, seq_len, N,2
        self.data_edge = torch.FloatTensor(edges_train)  
        #print(self.loc[0][0][0],self.vel[0][0][0])
        #print(self.loc[0][1][0],self.vel[0][1][0])
        
        
        
        self.tensor_data = torch.cat([self.loc,self.vel],dim=3)
        #print(self.tensor_data[0][0][0])
        #print(self.loc[0][0][0])
        #print(self.vel[0][0][0])
        
        torch.set_printoptions(precision=4,sci_mode=False)
       # print(self.tensor_data[0][0][0])
       # print(self.tensor_data[0][1][0])
        #pdb.set_trace()
        
        self.normalize_data()
        
        
        print(self.tensor_data.shape)
        print(self.data_edge.shape)
        
    def normalize_data(self):
        self.tensor_data[:,:,:,0:2] -= 8
        self.tensor_data[:,:,:,0:2] /= 8
        self.tensor_data[:,:,:,2:4] /= 8

    def __getitem__(self, item):
        sample = {'inputs': self.tensor_data[item],'edges':self.data_edge[item]}
        return sample

    def __len__(self):
        return self.num_sequences

    def get_sq_length(self):
        return self.num_frames

def prepare_flocking_dataset(args):
    all_data = np.load(args.data_path)
    all_labels = np.load(args.label_path)

    all_data = all_data[:, -args.obs_frames:, ...]

    all_data = torch.tensor(all_data)
    all_labels = torch.tensor(all_labels)

    print('data shape', all_data.shape)
    print('labels shape', all_labels.shape)

    N_agent = all_data.shape[2]
    N_time = all_data.shape[1]
    N_data = all_data.shape[0]
    N_feature_dim = all_data.shape[-1]
    dataset = torch.utils.data.TensorDataset(all_data, all_labels) # create your datset
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    torch.cuda.manual_seed_all(args.randomseed)
    torch.manual_seed(args.randomseed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
   # pdb.set_trace()
    # Parameters
    params_train = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              }
    params_test = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              }
    params_vali = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              }

    train_generator = torch.utils.data.DataLoader(train_dataset, **params_train)
    val_generator = torch.utils.data.DataLoader(val_dataset, **params_vali)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params_test)
    return train_generator, val_generator, test_generator,N_agent,N_time,N_data,N_feature_dim



class SmallSynthData(Dataset):
    def __init__(self, data_path, mode, params):
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'train':
            path = os.path.join(data_path, 'train_feats')
            edge_path = os.path.join(data_path, 'train_edges')
        elif self.mode == 'val':
            path = os.path.join(data_path, 'val_feats')
            edge_path = os.path.join(data_path, 'val_edges')
        elif self.mode == 'test':
            path = os.path.join(data_path, 'test_feats')
            edge_path = os.path.join(data_path, 'test_edges')
        self.feats = torch.load(path)
        self.edges = torch.load(edge_path)
        print("self.feats shape:",self.feats.shape)
        print("self.edges shape:",self.edges.shape)
        
        self.same_norm = params['same_data_norm']
        self.no_norm = params['no_data_norm']
        if not self.no_norm:
            self._normalize_data()

    def _normalize_data(self):
        train_data = torch.load(os.path.join(self.data_path, 'train_feats'))
        if self.same_norm:
            self.feat_max = train_data.max()
            self.feat_min = train_data.min()
            self.feats = (self.feats - self.feat_min)*2/(self.feat_max-self.feat_min) - 1
        else:
            self.loc_max = train_data[:, :, :, :2].max()
            self.loc_min = train_data[:, :, :, :2].min()
            self.vel_max = train_data[:, :, :, 2:].max()
            self.vel_min = train_data[:, :, :, 2:].min()
            self.feats[:,:,:, :2] = (self.feats[:,:,:,:2]-self.loc_min)*2/(self.loc_max - self.loc_min) - 1
            self.feats[:,:,:,2:] = (self.feats[:,:,:,2:]-self.vel_min)*2/(self.vel_max-self.vel_min)-1

    def unnormalize(self, data):
        if self.no_norm:
            return data
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
        else:
            result1 = (data[:, :, :, :2] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
            result2 = (data[:, :, :, 2:] + 1) * (self.vel_max - self.vel_min) / 2. + self.vel_min
            return np.concatenate([result1, result2], axis=-1)


    def __getitem__(self, idx):
        return {'inputs': self.feats[idx], 'edges':self.edges[idx]}

    def __len__(self):
        return len(self.feats)



if __name__ == '__main__':
    all_data = np.load('../../data/swarm/p_store_prey.npy')
    print(all_data.shape)
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output_dir', required=True)
    # parser.add_argument('--num_train', type=int, default=100)
    # parser.add_argument('--num_val', type=int, default=100)
    # parser.add_argument('--num_test', type=int, default=100)
    # parser.add_argument('--num_time_steps', type=int, default=50)
    # parser.add_argument('--pull_factor', type=float, default=0.1)
    # parser.add_argument('--push_factor', type=float, default=0.05)

    # args = parser.parse_args()
    # np.random.seed(1)
    # all_data = []
    # all_edges = []
    # num_sims = args.num_train + args.num_val + args.num_test
    # flip_count = 0
    # total_steps = 0
    # for sim in range(num_sims):
    #     p1_loc = np.random.uniform(-2, -1, size=(2))
    #     p1_vel = np.random.uniform(0.05, 0.1, size=(2))
    #     p2_loc = np.random.uniform(1, 2, size=(2))
    #     p2_vel = np.random.uniform(-0.05, -0.1, size=(2))
    #     p3_loc = np.random.uniform(-1, 1, size=(2))
    #     p3_vel = np.random.uniform(-0.05, 0.05, size=(2))

    #     current_feats = []
    #     current_edges = []
    #     for time_step in range(args.num_time_steps):
    #         current_edge = np.array([0,0,0,0,0,0])
    #         current_edges.append(current_edge)
    #         if np.linalg.norm(p3_loc - p1_loc) < 1:
    #             norm = np.linalg.norm(p3_loc - p1_loc)
    #             coef = 1 - norm
    #             dir_1 = (p3_loc - p1_loc)/norm
    #             p3_vel += args.push_factor*coef*dir_1
    #             current_edge[1] = 1
    #         if np.linalg.norm(p3_loc - p2_loc) < 1:
    #             norm = np.linalg.norm(p3_loc - p2_loc)
    #             coef = 1 - norm
    #             dir_2 = (p3_loc - p2_loc)/norm
    #             p3_vel += args.push_factor*coef*dir_2
    #             current_edge[3] = 1

    #         p1_loc += p1_vel
    #         p2_loc += p2_vel
    #         p3_loc += p3_vel
    #         p1_feat = np.concatenate([p1_loc, p1_vel])
    #         p2_feat = np.concatenate([p2_loc, p2_vel])
    #         p3_feat = np.concatenate([p3_loc, p3_vel])
    #         new_feat = np.stack([p1_feat, p2_feat, p3_feat])
    #         current_feats.append(new_feat)
    #     all_data.append(np.stack(current_feats))
    #     all_edges.append(np.stack(current_edges))
    # #pdb.set_trace()
    # all_data = np.stack(all_data)
    # train_data = torch.FloatTensor(all_data[:args.num_train])
    # val_data = torch.FloatTensor(all_data[args.num_train:args.num_train+args.num_val])
    # test_data = torch.FloatTensor(all_data[args.num_train+args.num_val:])
    # train_path = os.path.join(args.output_dir, 'train_feats')
    # torch.save(train_data, train_path)
    # val_path = os.path.join(args.output_dir, 'val_feats')
    # torch.save(val_data, val_path)
    # test_path = os.path.join(args.output_dir, 'test_feats')
    # torch.save(test_data, test_path)

    # train_edges = torch.FloatTensor(all_edges[:args.num_train])
    # val_edges = torch.FloatTensor(all_edges[args.num_train:args.num_train+args.num_val])
    # test_edges = torch.FloatTensor(all_edges[args.num_train+args.num_val:])
    # train_path = os.path.join(args.output_dir, 'train_edges')
    # torch.save(train_edges, train_path)
    # val_path = os.path.join(args.output_dir, 'val_edges')
    # torch.save(val_edges, val_path)
    # test_path = os.path.join(args.output_dir, 'test_edges')
    # torch.save(test_edges, test_path)
