import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import message_filters
import sys
from pathlib import Path
import time
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
publish_yolo_debug = False
if publish_yolo_debug:
    from yolo_test.utils.plots import Annotator,colors
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Header
from vas.msg import bbx

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)
        
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    
    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))
    
    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)
    
def xywh2xyxy(x, origin_h, origin_w, input_w, input_h):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression2(prediction, origin_h, origin_w, input_w, input_h, conf_thres=0.5, nms_thres=0.1):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
        prediction: detections, (x1, y1, x2, y2, conf, cls_id)
        origin_h: original image height
        origin_w: original image width
        conf_thres: a confidence threshold to filter detections
        nms_thres: a iou threshold to filter detections
    return:
        boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    # Get the boxes that score > CONF_THRESH

    #conf_thres = 0.01
    boxes = prediction[prediction[:, 4] >= conf_thres]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes[:, :4] = xywh2xyxy(boxes[:, :4], origin_h, origin_w, input_w, input_h )
    # clip the coordinates
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
    # Object confidence
    confs = boxes[:, 4]
    # Sort by the confs
    boxes = boxes[np.argsort(-confs)]
    # Perform non-maximum suppression
    keep_boxes = []
    while boxes.shape[0]:
        large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres

        label_match = np.abs(boxes[0, -1] - boxes[:, -1]) < 2e-1
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match

        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]
    boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    return boxes

class Param():
    def __init__(self,model='yolo_car', width=320, height=64, url='localhost:8001', 
                 confidence=0.85, nms=0.75, model_info=False, verbose=False, client_timeout=None):
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.confidence = confidence
        self.nms = nms
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout


def infer(client, model_name, input_data):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    outputs.append(grpcclient.InferRequestedOutput('output'))

    results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return results.as_numpy('output')

model_name_list = ['va', 'yolo_car']

class ImageSub():
    def __init__(self):
        rospy.init_node('yolo_node',anonymous=True)
        self.yolo_debug_pub = rospy.Publisher('/robot/yolo_debug', Image, queue_size=10)
        self.bbxpub = rospy.Publisher('/bbx', bbx, queue_size=10)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(20)
        self.omni_img = None
        self.img_msg_header = None
        self.yolo_params = Param()
        self.use_triton_server = True
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.yolo_params.url,
            verbose=self.yolo_params.verbose,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
        
        message_list = []
        omni_img_sub = message_filters.Subscriber('/robot/omni_img', Image)
        message_list.append(omni_img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        
        self.img_size = 320
        self.imgsz = [self.img_size,self.img_size]
        self.stride = 64
        self.pub_detection_debug = True

    def callback(self,data0):
        self.omni_img = self.bridge.imgmsg_to_cv2(data0, "bgr8")
        self.img_msg_header = data0.header
        
    def preprocess_input_image(self,raw_img):
        img = letterbox(raw_img, self.imgsz, stride=self.stride, auto=True)[0]
        # Convert
        #img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #cv2.cvtColor(img,cv2.) 
        img = img.reshape(1, img.shape[0], img.shape[1])
        img = np.ascontiguousarray(img)
        #self.img_size = check_img_size(self.img_size)
        img = np.expand_dims(img,0)
        img = img / 255
        #img = img.squeeze(0)
        #print("===========",img.shape)
        return img

    def preprocess_triton_input(self,raw_bgr_image, input_shape):
        """
        description: Preprocess an image before TRT YOLO inferencing.
                    Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.          
        param:
            raw_bgr_image: int8 numpy array of shape (img_h, img_w, 3)
            input_shape: a tuple of (H, W)
        return:
            image:  the processed image float32 numpy array of shape (3, H, W)
        """
        input_w, input_h = input_shape
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = input_w / w
        r_h = input_h / h
        if r_h > r_w:
            tw = input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((input_h - th) / 2)
            ty2 = input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = input_h
            tx1 = int((input_w - tw) / 2)
            tx2 = input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        return image


    def postprocess(self,output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, letter_box=False):
        """Postprocess TensorRT outputs.
        # Args
            output: list of detections with schema 
            [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            conf_th: confidence threshold
            letter_box: boolean, referring to _preprocess_yolo()
        # Returns
            list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
        """
        
        # Get the num of boxes detected
        # Here we use the first row of output in that batch_size = 1
        output = output[0]
        #num = int(output[0])
        # Reshape to a two dimentional ndarray
        #pred = np.reshape(output[1:], (-1, 6))[:num, :]

        # Do nms
        #print(output.shape)
        boxes = non_max_suppression2(output, origin_h, origin_w, input_shape[0], input_shape[1], conf_thres=conf_th, nms_thres=nms_threshold)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5].astype(int) if len(boxes) else np.array([])
            
        detected_objects = []
        for box, score, label in zip(result_boxes, result_scores, result_classid):
            detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], origin_h, origin_w))
        return detected_objects


    def yolo_process(self):
        agnostic_nms = False
        hide_labels = False
        hide_conf = False
        use_triton_server = True
        while not rospy.is_shutdown():
            if self.omni_img is not None:
                # Padded resize
                xy_array = []
                if True:
                    inputs = []
                    outputs = []
                    T1 = time.time()
                    input_image_buffer = self.preprocess_input_image(self.omni_img).astype(np.float32)
                    inputs.append(grpcclient.InferInput('input', input_image_buffer.shape, "FP32"))
                    outputs.append(grpcclient.InferRequestedOutput('output'))
                    inputs[0].set_data_from_numpy(input_image_buffer)
                    T2 = time.time()
                    print('pre process time: %s ms' % ((T2 - T1)*1000))

                    T1 = time.time()
                    result = self.triton_client.infer(model_name=self.yolo_params.model,
                                                        inputs=inputs,
                                                        outputs=outputs,
                                                        client_timeout=self.yolo_params.client_timeout)
                    T2 = time.time()
                    result = result.as_numpy('output')
                    print('infer time: %s ms' % ((T2 - T1)*1000))
                    print(f"{result.shape}")
                    
                    T1 = time.time()
                    detected_objects = self.postprocess(result, self.omni_img.shape[1], self.omni_img.shape[0], \
                                                   [self.yolo_params.width, self.yolo_params.height],
                                                   self.yolo_params.confidence,self.yolo_params.nms)
                    T2 = time.time()
                    print('postprocess time: %s ms' % ((T2 - T1)*1000))
                    print('\n')
                    for o in detected_objects:
                        x0,y0,x1,y1 = o.box()
                        xy_array.append(x0)
                        xy_array.append(y0)
                        xy_array.append(x1)
                        xy_array.append(y1)
                    
                    #### publish detection result for debug
                    if publish_yolo_debug:
                        annotator = Annotator(self.omni_img, line_width=3, example='car')
                        if len(detected_objects):
                            img = input_image_buffer
                            for o in detected_objects:
                                xyxy = o.box()
                                # rospy.loginfo(f"img {img.shape}")
                                # rospy.loginfo(f"self.omni_img {self.omni_img.shape}")
                                # rospy.loginfo(f"xyxy {xyxy}")
                                label = 'car'
                                annotator.box_label(xyxy, label, color=colors(0, True))
                        im0 = annotator.result()
                        if im0 is not None:
                            msg = self.bridge.cv2_to_imgmsg(im0,"bgr8")
                            self.yolo_debug_pub.publish(msg)

                array_msg = bbx()
                header = Header()
                header.stamp = self.img_msg_header.stamp

                layout = MultiArrayLayout()
                layout.dim.append(MultiArrayDimension())
                layout.dim[0].label = "rows"
                layout.dim[0].size = len(detected_objects)
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