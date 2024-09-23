import numpy as np
import torch
import sys
from pathlib import Path
import os
import cv2
import onnx
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # YOLOv5 root directory
ROOT_YOLO = os.path.join(str(FILE.parents[1]), './yolo_test')
if str(ROOT_YOLO) not in sys.path:
    sys.path.append(str(ROOT_YOLO))  # add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from cfg_vatt import config_train as cfg
from data_utils import *
from yolo_test.models.experimental import attempt_load
from yolo_test.utils.general import check_img_size
from yolo_test.utils.augmentations import letterbox

def preprocess_input_image(raw_img,stride,img_size = 640,ch = 3):
    img = letterbox(raw_img, img_size, stride=stride, auto=True)[0]
    if ch == 3:
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    elif ch == 1:
        img = img.reshape(1, img.shape[0], img.shape[1])
    else:
        print("wrong image channel:",ch)
        exit()
    
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().cuda()
    img_size = check_img_size(img_size)
    img = img[None]
    img /= 255
    return img
    
if __name__ == "__main__":
    img_size = 320
    imgsz = [img_size,img_size]
    stride = 64
    device = torch.device("cpu")
    weights_file = cfg['yolo_dir']
    model = attempt_load([weights_file], device=device)
    stride = int(model.stride.max())  # model stride
    yolo_model = model.eval()
    print(yolo_model)
    print("load yolo ok!!!")

    raw_img = cv2.imread('yolo_test/2319.png',cv2.IMREAD_GRAYSCALE)
    print(raw_img.shape)
    if len(raw_img.shape) == 2 or raw_img.shape[2] == 1:
        ch = 1
    else:
        ch = 3
    
    print("image channel:",ch)
    stride = int(yolo_model.stride.max())  # model stride
    img = preprocess_input_image(raw_img,stride,img_size,ch)

    print(f'image shape {img.shape} stride {stride}')
    file_name = 'onnx/yolo_car_mono_close_dynamic.onnx'
    net = yolo_model.to(torch.float32)
    img = img.to(torch.float32).cpu()
    net.eval()
    
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(net.cpu(),img,file_name,verbose=False,opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes)
    model_onnx = onnx.load(file_name)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, file_name)
    print("save onnx ok!!!")