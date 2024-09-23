import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import torch
import cv2
import sys
from pathlib import Path
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # YOLOv5 root directory
ROOT_YOLO = os.path.join(str(FILE.parents[1]), './yolo_test')
if str(ROOT_YOLO) not in sys.path:
    sys.path.append(str(ROOT_YOLO))  # add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from cfg_vatt import config_train as cfg
import pdb
from data_utils import *
from yolo_test.models.experimental import attempt_load
from yolo_test.utils.general import check_img_size,non_max_suppression,scale_boxes
from yolo_test.utils.augmentations import letterbox
from yolo_test.utils.plots import Annotator,colors
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Float32MultiArray,Header
from vas.msg import bbx

class ImageSub():
    def __init__(self):
        rospy.init_node('yolo_node',anonymous=True)
        self.yolo_debug_pub = rospy.Publisher('/robot/yolo_debug', Image, queue_size=10)
        self.bbxpub = rospy.Publisher('/bbx', bbx, queue_size=10)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(1000)
        self.omni_img = None
        self.img_msg_header = None
        message_list = []
        omni_img_sub = message_filters.Subscriber('/robot/omni_img', Image)
        message_list.append(omni_img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        ###############################################
        self.img_size = 320
        self.imgsz = [self.img_size,self.img_size]
        self.stride = 64
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        weights_file = cfg['yolo_dir']
        model = attempt_load([weights_file], device=self.device)
        self.stride = int(model.stride.max())  # model stride
        self.yolo_model = model.eval()
        print(self.yolo_model)
        print("load yolo ok!!!")
        ##########################################

    def callback(self,data0):
        #self.omni_img = self.bridge.imgmsg_to_cv2(data0, "bgr8")
        self.omni_img = self.bridge.imgmsg_to_cv2(data0, "mono8")
        self.img_msg_header = data0.header
        
    def preprocess_input_image(self,raw_img):
        img = letterbox(raw_img, self.img_size, stride=self.stride, auto=True)[0]
        # Convert
        #img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
       # pdb.set_trace()
        
        img = img.reshape(1, img.shape[0], img.shape[1])
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().cuda()
        self.img_size = check_img_size(self.img_size)
        img = img[None]
        img /= 255
        return img

    def yolo_process(self):
        conf_thres = 0.85 # confidence threshold
        iou_thres = 0.75  # NMS IOU threshold
        max_det = 4  # maximum detections per image
        classes = 0
        agnostic_nms = False
        hide_labels = False
        hide_conf = False
        names = self.yolo_model.names if hasattr(self.yolo_model, 'module') else self.yolo_model.names  # get class names
        img_cnt = 0
        while not rospy.is_shutdown():
            if self.omni_img is not None:
                # Padded resize
                self.stride = int(self.yolo_model.stride.max())  # model stride
                rospy.loginfo(f"self.omni_img {self.omni_img.shape}")
                
                img = self.preprocess_input_image(self.omni_img)
                rospy.loginfo(f"model input image shape {img.shape}")
                
                pred = self.yolo_model(img, False, False)[0]
                #rospy.loginfo(f"conf_thres {conf_thres} iou_thres {iou_thres}")
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                
                # Process predictions
                im0 = self.omni_img.copy() 
                #rospy.loginfo(f"img shape {img.shape}")
                #rospy.loginfo(f"im0 shape {im0.shape}")
                xy_array = []
                im0_flip = cv2.flip(im0,-1)
                for i, det in enumerate(pred):  # per image
                    annotator = Annotator(im0_flip, line_width=3, example=str('car'))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        # rospy.loginfo(f"img {img.shape}")
                        #rospy.loginfo(f"11111111111111 img0 {im0.shape}")
                        # rospy.loginfo(f"det[:, :4] {det[:, :4]}")
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            ########### flip xyxy ##########
                            xyxy_display = copy.deepcopy(xyxy)
                            xyxy_display[0] = im0.shape[1] - xyxy_display[0] -1
                            xyxy_display[2] = im0.shape[1] - xyxy_display[2] -1
                            xyxy_display[1] = im0.shape[0] - xyxy_display[1] -1
                            xyxy_display[3] = im0.shape[0] - xyxy_display[3] -1
                            ################################
                            annotator.box_label(xyxy_display, label, color=colors(c, True))
                            for i in xyxy:
                                i = i.cpu().detach().float().item()
                                xy_array.append(i)
                im0_flip = annotator.result()
                if im0_flip is not None:
                    #cv2.imwrite(f"detection_figure/{img_cnt}.png",im0_flip)
                    img_cnt = img_cnt + 1
                    msg = self.bridge.cv2_to_imgmsg(im0_flip,"mono8")
                    self.yolo_debug_pub.publish(msg)
                array_msg = bbx()
                header = Header()
                header.stamp = self.img_msg_header.stamp

                layout = MultiArrayLayout()
                layout.dim.append(MultiArrayDimension())
                layout.dim[0].label = "rows"
                layout.dim[0].size = len(det)
                layout.dim.append(MultiArrayDimension())
                layout.dim[1].label = "cols"
                layout.dim[1].size = 4
                array_msg.layout = layout

                array_msg.data = xy_array
                array_msg.header = header
                self.bbxpub.publish(array_msg)

            self.loop_rate.sleep()

if __name__ == '__main__':
    img_sub = ImageSub()
    img_sub.yolo_process()