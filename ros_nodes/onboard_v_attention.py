import os
import platform
ONBOARD = False
if platform.machine() == 'aarch64':
    print("Running on Jetson")
    ONBOARD = True
elif platform.machine() == 'x86_64':
    print("Running on x86")
else:
    raise Exception("Unknown platform")

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
import sys
sys.path.append("..")
sys.path.append(".")
from cfg_vatt import config_train as cfg
from data_utils import convert_np,normalize_image
if not ONBOARD:
    import torch
    from models.va_net import VattNet
    #from data_utils import convert_np,normalize_image

import time,threading
import json
if ONBOARD:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
from data_utils import preprocess,normalize_array

class Param():
    def __init__(self,model='va', width=410, height=(135 //2 + 1), url='localhost:8001', model_info=False, verbose=False, client_timeout=None):
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

class ImageSub():
    def __init__(self):
        rospy.init_node('v_attention_node',anonymous=True)
        if not ONBOARD:
            self.v_att_debug_pub = rospy.Publisher('/robot/v_att_debug', Image, queue_size=10)
        self.v_att_pub = rospy.Publisher('/robot/v_att', Image, queue_size=10)
        with open('cali_file/params.json', 'r') as f:
            params = json.load(f)
        print("params:",params)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(30)
        self.rz_scale = params['vatt_rz']
        self.omni_img = None
        self.bbx = None
        self.detected_tar_num = 0
        self.va_param = Param()
        self.lock = threading.Lock()

        self.current_time = None
        self.processed_frame = 0

        message_list = []
        omni_img_sub = message_filters.Subscriber('/robot/omni_img', Image)
        message_list.append(omni_img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        if ONBOARD:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.va_param.url,
                verbose=self.va_param.verbose,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None)
        else:
            ########### read v attention model #########
            #[32,64,128,256,512]
            channel = cfg['va_channel']
            bilinear = cfg['bilinear']
            self.v_attention_model = VattNet(n_channels=1, n_classes=1,channel = channel
                                             , bilinear=bilinear)
            cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:0" if cuda else "cpu")
            self.v_attention_model.to(device=self.device)
            model_dir = cfg['model_dir']
            state_dict = torch.load(model_dir, map_location=self.device)
            self.v_attention_model.load_state_dict(state_dict)
            self.v_attention_model.eval()
            #############################################

    def callback(self,data0):
        self.lock.acquire()
        try:
            self.current_time = data0.header.stamp
            self.omni_img = self.bridge.imgmsg_to_cv2(data0, "bgr8")
        finally:
            self.lock.release()

    def dummy_process(self):
        while not rospy.is_shutdown():
            print(111)
            self.loop_rate.sleep()

    def v_attention_process(self):
        img_cnt = 0
        
        while not rospy.is_shutdown():
            if self.omni_img is not None:
                if ONBOARD:
                    T1 = time.time()
                    inputs = []
                    outputs = []
                    self.lock.acquire()
                    try:
                        input_image_buffer = preprocess(self.omni_img,scale = self.rz_scale).astype(np.float32)
                    finally:
                        self.lock.release()
                    #rospy.loginfo(f"input_image_buffer shape {input_image_buffer.shape}")
                    
                    ### [1, 1, self.va_param.height, self.va_param.width]

                    input_image_buffer = np.expand_dims(input_image_buffer,axis=0)
                    #rospy.loginfo(f'input_image_buffer shape {input_image_buffer.shape}')
                    input_image_buffer = np.expand_dims(input_image_buffer,axis=0)
                    
                    ###  input_image_buffer = np.transpose(input_image_buffer,[0,3,1,2])
                    
                    inputs.append(grpcclient.InferInput('images', input_image_buffer.shape, "FP32"))
                    outputs.append(grpcclient.InferRequestedOutput('v_atten_maps'))
                    
                    inputs[0].set_data_from_numpy(input_image_buffer)
                    #rospy.loginfo(f'input_image_buffer shape {input_image_buffer.shape}')
                    
                    T2 = time.time()

                    rospy.loginfo('pre process time:%s ms' % ((T2 - T1)*1000))
                    T1 = time.time()
                    result = self.triton_client.infer(model_name=self.va_param.model,
                                                        inputs=inputs,
                                                        outputs=outputs,
                                                        client_timeout=self.va_param.client_timeout)
                    result = result.as_numpy('v_atten_maps')
                    T2 = time.time()
                    rospy.loginfo('infer time:%s ms\n' % ((T2 - T1)*1000))
                    
                    T1 = time.time()
                    mask = sigmoid(result)
                    T2 = time.time()
                    rospy.loginfo('sigmoid time:%s ms' % ((T2 - T1)*1000))
                    self.lock.acquire()
                    try:
                        mask = np.squeeze(mask,axis=0)
                        mask = np.squeeze(mask,axis=0)
                        
                        ni = normalize_image(mask)
                        # print("mask ",mask.shape)
                        # cv2.imshow("ni",ni)
                        # cv2.waitKey(1)
                        if ni is not None:
                            msg = self.bridge.cv2_to_imgmsg(ni,"mono8")
                            msg.header.stamp = self.current_time
                    finally:
                        self.lock.release()
                    if ni is not None:
                        self.v_att_pub.publish(msg)
                    #avg_probs = normalize_array(avg_probs)
                else:
                    self.lock.acquire()
                    try:
                        img = torch.from_numpy(preprocess(self.omni_img,scale = self.rz_scale))
                        #rospy.loginfo(f"img shape after preprocess {img.shape},{self.omni_img.shape}")
                        #img = img.unsqueeze(0).permute(0,3,1,2).cuda().float()
                        img = img.unsqueeze(0).unsqueeze(0).cuda().float()
                        predicted = self.v_attention_model(img)
                        mask = torch.log_softmax(predicted,dim=-1)
                        
                        temp_mask = normalize_image(convert_np(mask)).astype(np.uint8)
                        #temp_mask = cv2.resize(temp_mask,dsize=None,fx=1/self.rz_scale,fy=1/self.rz_scale,interpolation = cv2.INTER_NEAREST)
                        temp_mask = cv2.resize(temp_mask,dsize=None,fx=1/self.rz_scale,fy=1/self.rz_scale,interpolation = cv2.INTER_LINEAR)
                        
                        heat_map_trans = cv2.applyColorMap(temp_mask, cv2.COLORMAP_JET)
                        overlapping = cv2.addWeighted(self.omni_img, 0.5, heat_map_trans, 0.5, 0)
                        
                        ni = normalize_image(convert_np(mask))
                        rospy.loginfo(f"ni {ni.shape}")

                        if overlapping is not None:
                            overlapping = cv2.flip(overlapping,-1)
                            msg = self.bridge.cv2_to_imgmsg(overlapping,"bgr8")
                            msg.header.stamp = self.current_time
                        if ni is not None:
                            msg_att = self.bridge.cv2_to_imgmsg(ni,"mono8")
                            msg_att.header.stamp = self.current_time
                            self.v_att_pub
                    finally:
                        self.lock.release()
                        if overlapping is not None:
                            self.v_att_debug_pub.publish(msg)
                        if ni is not None:
                            self.v_att_pub.publish(msg_att)
            else:
                rospy.loginfo(f"No image!!")
            self.loop_rate.sleep()

if __name__ == '__main__':
    img_sub = ImageSub()
    img_sub.v_attention_process()
    #img_sub.dummy_process()