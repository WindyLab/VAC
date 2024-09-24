# Collective Visual Attention Clone via Neural Interaction Graph Prediction
This repository contains code for our paper "Collective Visual Attention Clone via Neural Interaction Graph Prediction".

## Environment
#### Offline
The code runs on Ubuntu 20.04 with ros noetic.     
#### Onboard
The robot uses a Jetson Xavier NX with JetPack 5.1.3.

#### Setup the Environment
To setup the offline python environment for model training and offline running:     

```bash
conda env create -f environment.yaml
conda activate vas
```

## Data Preparation
The data and checkpoints can be downloaded here:<a id="data-id"></a>                 
https://pan.baidu.com/s/1GiTkiQ3HAEgBWp0ixGFDFw?pwd=pk34                    
password: pk34          
We provide ros bags in Gazebo and real-world environment.                     

## Build CPP ROS Nodes
Some nodes are implemented in C++, so go to the vas_ws to build them:         
```bash
cd vas_ws
catkin_make
```
## Offline Running
Before running offline, make sure that the required Conda environment is properly set up. Once configured, download the check points and unzip it in the root folder.
```bash
unzip checkpoints.zip -d ./
```
then you can publish the data and start the all-in-one script:
```bash
rosbag play PATH_TO_THE_BAG
bash script/off_start.sh
```
If you prefer to run each module separately, refer to the respective modules in the [off_start.sh](./script/off_start.sh) script. The `host_id` parameter in the `image_concatenator` node corresponds to the ID of each robot. For data in Gazebo simulations, set `host_id=999`.              

The offline running code demonstrates the overall workflow, including visual detection results and velocity commands. In real-world or Gazebo simulation experiments, the robot will respond to the velocity commands and move to exhibit collective behavior.            

## Onboard Running
To run the code on a real robot, all network models must be converted to [TensorRT](https://github.com/NVIDIA/TensorRT) in advance. For tutorials on TensorRT and Triton, please refer to Nvidia's official resources: [TensorRT](https://github.com/NVIDIA/TensorRT) and [Triton](https://github.com/triton-inference-server). These details will not be covered in this project.

To execute the code onboard, first start the Triton server. Then, modify the specified lines in [off_start](./script/off_start.sh) and run the script.

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
The training and inference code of our GVAE model is in `gvae` folder. Use the scripts in [`run_scripts`](./gvae/run_scripts) to train the model. A training example data is also provided in the [Data Preparation](#data-id). You can unzip and copy it to the `gvae` folder.


## Acknowledgement
* Our GVAE takes graph network operators in [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
* Our baselines of interaction graph prediction comparison are from [GST](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph), [NRI](https://github.com/ethanfetaya/NRI),[IMMA](https://github.com/sunfanyunn/IMMA),[dNRI](https://github.com/cgraber/cvpr_dNRI). The graph attention network (GAT) used in the comparison takes the implementation in [IMMA](https://github.com/sunfanyunn/IMMA). Our model also refer to the above implementations. We extend our gratitude to the authors of the above projects for their open-source contributions.
* Our json loading in [JSON](./vas_ws/src/vas/src/json.hpp) is borrowed from [json for modern C++](https://github.com/nlohmann/json).
* Our [`YOLOv5`](./yolo_test) running code is adapted from the official code of [YOLOv5](https://github.com/ultralytics/yolov5).
* The onboard code utilizes the acceleration and scheduling of [TensorRT](https://github.com/NVIDIA/TensorRT) and [Triton](https://github.com/triton-inference-server).
* The robot swarm system for simulation and real-world experiment is based on our previous work. More details can be found at [Omnibot](https://shiyuzhao.westlake.edu.cn/2024ICCA_Omnibot.pdf).

## Others
This project is licensed under the [MIT License](./LICENSE).            
If you have any questions, please contact likai [at] westlake [dot] edu [dot] cn

