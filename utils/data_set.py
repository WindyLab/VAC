import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import pdb
import cv2
import logging
from torchvision import transforms
import os
import yaml
import matplotlib.pyplot as plt

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
    return loc_train, vel_train, edge_train

class VicsekData(Dataset):
    def __init__(self,fea,edge,num_agent,len_sequence,num_sequence):

        super(VicsekData, self).__init__()
        self.num_agent = num_agent
        self.num_frames = len_sequence
        self.num_sequences = num_sequence
        # loc_train = np.transpose(loc_train,[0,2,1,3])
        # vel_train = np.transpose(vel_train,[0,2,1,3])
      #  pdb.set_trace()
        self.data_edge = edge
        self.tensor_data = fea
        print(self.tensor_data.shape)
        print(self.data_edge.shape)

    def __getitem__(self, item):
        sample = {'inputs': self.tensor_data[item],'edges':self.data_edge[item]}
        return sample

    def __len__(self):
        return self.num_sequences

    def get_sq_length(self):
        return self.num_frames

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, x_seq_len=5, y_seq_len=5):
        self.X = data
        self.y = data
        self.x_seq_len = x_seq_len
        self.y_seq_len = y_seq_len

    def __len__(self):
        return self.X.shape[1] - (self.x_seq_len+self.y_seq_len)

    def __getitem__(self, index):
        if index > self.X.shape[1] - (self.x_seq_len+self.y_seq_len):
          index = self.X.shape[1] - (self.x_seq_len+self.y_seq_len)
        return (self.X[:,index:index+self.x_seq_len,:], self.y[:,index+self.x_seq_len:index+self.x_seq_len+self.y_seq_len,:])
    
class VSwarmDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 hms_dir: str,
                 scale: float = 1.0,
                 ch: int = 3,
                 motion: bool = False):
        self.images_dir = images_dir
        self.hms_dir = hms_dir
        self.motion = motion
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.img_files = os.listdir(self.images_dir)
        self.hms_files = os.listdir(self.hms_dir)
        
        self.img_files.sort(key=lambda x: float(x.split(".")[0]))
        self.hms_files.sort(key=lambda x: float(x.split(".")[0]))
        print(self.img_files[0:10])
        print(self.hms_files[0:10])

        self.num_hms = len(self.hms_files)
        self.num_img = len(self.img_files)
        self.ch = ch
        assert  self.num_hms == self.num_img
        logging.info(f'Creating dataset with {self.num_hms} / {self.num_img} examples')

    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        #read original image
        img_name = self.images_dir +'/' +  self.img_files[idx]
        orig_img = None
        img = None
        if not self.motion:
            if self.ch == 1:
                orig_img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
            else:
                orig_img = cv2.imread(img_name,cv2.IMREAD_UNCHANGED)
        else:
            orig_img = np.load(img_name,allow_pickle=True)
        img = cv2.resize(orig_img,dsize=None,fx=self.scale,fy=self.scale)
        if len(img.shape) == 2:
            img = np.expand_dims(img,axis=2) #motion_flow = np.expand_dims(motion_flow, axis=2)
        if self.ch == 3:
            if np.random.randint(0,2) == 0:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if np.random.randint(0,10) == 0:
                img = self.preprocess(img)

        if np.random.randint(0,2) == 1:
            img = self.add_noise(img)

        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.to(torch.float32)

        ######  read heatmap ############
        hms_name = self.hms_dir + '/' + self.hms_files[idx]
        heatmap = np.load(hms_name,allow_pickle=True)
       # pdb.set_trace()
        heatmap_resize = cv2.resize(heatmap,dsize=None,fx=self.scale,fy=self.scale)
        #print(img_name,heatmap_resize.shape,img.shape,self.scale)

        #plt.matshow(heatmap)

        # plt.matshow(img[:,:,0])
        # #plt.matshow(img[:,:,1])
        #plt.show()
        
        # plt.waitforbuttonpress()
        #plt.close()

        heatmap_resize = torch.tensor(heatmap_resize, dtype=torch.float32) / 255.0
        heatmap_resize = heatmap_resize
        return img_tensor,heatmap_resize

    def preprocess(self, data):
        # random hue and saturation
        img_hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.float64)
        delta = (np.random.random() * 2 - 1) * 0.2
        img_hsv[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        img_hsv[:, :, 1] *= delta_sature
        img_hsv[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        img_hsv = img_hsv * 255.0
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        # adjust brightness
        img_rgb = img_rgb.astype(np.float64)
        delta = (np.random.random() * 2 - 1) * 0.3
        img_rgb += delta
        return img_rgb.astype(np.uint8)

    def add_noise(self,img):
        row,col,ch = img.shape
        gauss = np.random.normal(0,0.9,(row,col,ch))
        img = img.astype(np.float64) + gauss
        return img.astype(np.uint8)


if __name__ == '__main__':
    data_path = 'gazebo_motion_data_0/data/'
    vattention_label_path = 'gazebo_motion_data_0/att_labels/'
    cur_data_set = VSwarmDataset(data_path,
                                 vattention_label_path,scale = 0.25,motion=True)
    
    for idx, data in enumerate(cur_data_set):
        print(idx)