# Collective Visual Attention Clone via Neural Interaction Graph Prediction
This repository contains code for our paper "Collective Visual Attention Clone via Neural Interaction Graph Prediction".

## Environment
#### Offline
The code runs on Ubuntu 20.04 with ros noetic.     
#### Onboard
The robot uses a Jetson Xavier NX with JetPack 5.1.3.

#### Setup the Environment
To setup the offline python environment for model training and offline running.

```bash
conda env create -f environment.yaml
conda activate vas
```

## Build CPP ROS Nodes
Some nodes are implemented in C++, so go to the vas_ws to build them.
```bash
cd vas_ws
catkin_make
```

## Data Preparation
The rosbag can be downloaded here:                    
We provide bags in Gazebo and real-world environment.                     

## Offline Running
The offline running requires the conda environment, make sure you configure it before starting.
To publish the data and start all-in-one running script:    
```bash
rosbag play PATH_TO_THE_BAG
bash script/off_start.sh
```
If you prefer running each module separately, check the corresponding modules in off_start.sh.       
The host_id parameter in the image_concatenator node refers to the id of each robot. For data in Gazebo simulation, the host_id=999. The offline running code will show the general workflow, visual detection results, and velocity command. In the real-world or gazebo simulation experiment, the robot responses to the velocity command and moves to form collective behavior.

## Onboard Running
All models must be converted to [TensorRT](https://github.com/NVIDIA/TensorRT) in advance.      
Start the triton-server, then change the following lines in [off_start.sh](./script/off_start.sh)
```bash
python3 ros_nodes/ob_detection.py  ->  python3 ros_nodes/onboard_detection.py
```

## Training
#### Visual Attention
The training and testing codes are in va_model folder.    
```bash
cd va_model
conda activate vas
python3 train_vatt.py
```

#### Yolo
We use the official code to train the YOLOv5.
The input channel is modified to 1 for computational efficiency.
The yolo_test folder contains the code borrowed from YOLOv5 for offline running,
onnx conversion and post-process.

## Neural Interaction Graph Prediction
The training and inference code of our GVAE model is in gvae folder. Use the scripts in [`run_scripts`](./gvae/run_scripts) to train the model.


## Acknowledgement
* Our GVAE takes graph network operators in [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
* Our baselines of interaction graph prediction comparison are from [GST](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph), [NRI](https://github.com/ethanfetaya/NRI),[IMMA](https://github.com/sunfanyunn/IMMA),[dNRI](https://github.com/cgraber/cvpr_dNRI). The graph attention network (GAT) used in the comparison takes the implementation in [IMMA](https://github.com/sunfanyunn/IMMA). Our model also refer to the above implementations. We extend our gratitude to the authors of the above projects for their open-source contributions.
* Our json loading in [JSON](./vas_ws/src/vas/src/json.hpp) is borrowed from [json for modern C++](https://github.com/nlohmann/json).
* Our [`YOLOv5`](./yolo_test) running code is adapted from the official code of [YOLOv5](https://github.com/ultralytics/yolov5).
* The onboard code utilizes the acceleration and scheduling of [TensorRT](https://github.com/NVIDIA/TensorRT) and [Triton](https://github.com/triton-inference-server).


