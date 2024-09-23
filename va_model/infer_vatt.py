import torch
import cv2
from cfg_vatt import config_train as cfg
import numpy as np
from models.va_net import VattNet
from matplotlib import pyplot as plt
import pdb
import json

from data_utils import *
import onnx
import time

def predict_img_test(net,img,device):
    net.eval()
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        start_time = time.time()
        output = net(img)
        mask = torch.sigmoid(output.cpu())
        end_time = time.time()
        time_difference = end_time - start_time
        print("time_difference:",time_difference)
        print("1111111111111:",output.shape,img.shape)
        out_show = convert_np(output)
        mask_show = convert_np(mask)
        normalized_mask = cv2.normalize(mask_show, None, 0, 255, cv2.NORM_MINMAX)
        print("out_show:",out_show.shape)
        plt.matshow(out_show)
        img_show = img.squeeze(0).permute(1,2,0)
        plt.matshow(img_show.detach().cpu().numpy())
        plt.matshow(normalized_mask)
        plt.show()

if __name__ == '__main__':
    ######################### Parameters ####################################
    para_path = 'cali_file/params.json'
    with open(para_path,encoding='utf-8-sig', errors='ignore') as f:
        parameters = json.load(f, strict=False)
        print(parameters)
    scale = parameters['vatt_rz']
    ch = 1
    ########################################################################
    ######################### LOAD MODEL####################################
    channel = cfg['va_channel']
    bilinear = cfg['bilinear']
    net = VattNet(n_channels=ch, n_classes=1,channel=channel,bilinear=bilinear)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu") 
   # logging.info(f'Using device {device}')
    net.to(device=device)
    model_dir = cfg['model_dir']
    state_dict = torch.load(model_dir, map_location=device)
    net.load_state_dict(state_dict)
    #######################################################################
    img_dir = 'real_world_data_0/data/00048.png'
    img = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(preprocess(img,scale=scale))
    #pdb.set_trace()
    
    ### input shape [batch=1, channel,height,width]
    if ch == 3:
        img = img.unsqueeze(0).permute(0,3,1,2).to(torch.float32)
    else:
        img = img.unsqueeze(0).to(torch.float32)
        img = img.unsqueeze(0)
    print(f'image shape {img.shape}')
    
    cnt = 0
    # while cnt < 1000:
    #     predict_img_test(net,img,device)
    #     cnt = cnt + 1
    
    ############### export onnx ###############
    file_name = 'onnx/v_attention_mono_nano2.onnx'
    net = net.to(torch.float32)
    net.eval()
    dynamic_axes = {
        'images': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
        'v_atten_maps': {0: 'batch_size'}
    }
    torch.onnx.export(net.cpu(),img,file_name,verbose=False,opset_version=12,
        input_names=['images'],
        output_names=['v_atten_maps'],
        dynamic_axes=dynamic_axes)
    
    ## constant folding may not work with opset_version > 11
    model_onnx = onnx.load(file_name)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, file_name)
    print("onnx save ok!")
