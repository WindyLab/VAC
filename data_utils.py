import numpy as np
import yaml
import rosbag
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import cv2
import tf
import ros
import rospy
import shutil
import os
import pdb
import math
import copy
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import ModelState
from img_concatenater import ImageConcatenater
from cam_para_manager import *

class Car():
    def __init__(self):
        super(Car, self).__init__()
        self.image_vec = []
        self.odom_vec = []
        self.gazebo_state = []
        self.image_time_vec = []
        self.host_id = -1
        self.raw_img_list_list = []

    def add_raw_image(self,raw_img_list):
        """
        add raw image
        """
        self.raw_img_list_list.append(raw_img_list)

    def add_image(self,image):
        """
        add omnidirectional image
        """
        self.image_vec.append(image)

    def add_image_time(self,img_time):
        """
        add image time
        """
        self.image_time_vec.append(img_time)

    def add_odom(self,odom):
        """
        add odom, one image -> odom
        """
        self.odom_vec.append(odom)
        
    def add_gazebo_state(self,gazebo_state):
        """
        add gazebo_state, one image -> gazebo_state
        """
        self.gazebo_state.append(gazebo_state)

    def get_data_num(self):
        assert len(self.image_vec) == len(self.odom_vec)
        return len(self.image_vec)
    
def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

def generate_random_training_idx(total_data_num, training_num):
    ret = getRandomIndex(total_data_num,training_num)
    ret.sort()
    return ret

def generate_even_training_idx(total_data_num, interval,start_from = 0):
    ret = range(start_from, total_data_num, interval) 
    return ret

def gaussian_k(x0,y0,sigma, width, height):
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
 
def generate_hm(height, width ,landmarks,s=3):
    Nlandmarks = landmarks.shape[0]
    hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1,-1]):
            
            hm[:,:,i] = gaussian_k(landmarks[i][0],
                                    landmarks[i][1],
                                    s,width,height)
        else:
            hm[:,:,i] = np.zeros((height,width))
    return hm

def convert_time(ros_time):
    t = ros_time.secs * 1e09 + ros_time.nsecs
    return t

def get_gazebo_state(gazebo_state_vec, t):
    def compare_timestamp(gazebo_state):
        return (gazebo_state[0] - t).to_sec()
    left = 0
    right = len(gazebo_state_vec) - 1
    closest_state = None
    while left <= right:
        mid = (left + right) // 2
        diff = compare_timestamp(gazebo_state_vec[mid])
        if diff == 0:
            return gazebo_state_vec[mid]
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
        if closest_state is None or abs(diff) < abs(compare_timestamp(closest_state)):
            closest_state = gazebo_state_vec[mid]
    return closest_state

def get_odom(odom_vec, t):
    def compare_timestamp(odom):
        return (odom.header.stamp - t).to_sec()
    left = 0
    right = len(odom_vec) - 1
    closest_odom = None
    while left <= right:
        mid = (left + right) // 2
        diff = compare_timestamp(odom_vec[mid])
        if diff == 0:
            return odom_vec[mid]
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
        if closest_odom is None or abs(diff) < abs(compare_timestamp(closest_odom)):
            closest_odom = odom_vec[mid]
    return closest_odom

def get_img_time(img_time_vec, t):
    def compare_timestamp(img_time):
        return (img_time - t).to_sec()
    left = 0
    right = len(img_time_vec) - 1
    closest_img_t = None
    while left <= right:
        mid = (left + right) // 2
        diff = compare_timestamp(img_time_vec[mid])
        if diff == 0:
            return img_time_vec[mid]
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
        if closest_img_t is None or abs(diff) < abs(compare_timestamp(closest_img_t)):
            closest_img_t = img_time_vec[mid]
    return closest_img_t


def read_odom(bag_dir,car_id):
    bag = rosbag.Bag(bag_dir)
    topic_info = bag.get_type_and_topic_info()
    print(topic_info)
    print(bag_dir)
    odom_car = []
    gazebostate_car = []
    ######## use robot odom  ##############
    for topic, msg, t in bag.read_messages(topics=['/robot/odom']):
        #print(f"Received message on topic {topic} at time {t}")
        #print("odom time diff:",convert_time(t - msg.header.stamp)/1e06)
        #print(msg.header.stamp)
        msg_offset = msg
        msg_offset.header.stamp = t
        odom_car.append(msg_offset)

   # pdb.set_trace()
    ######## use gazebo state  #############
    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
        print(t)
        pose_id = msg.name.index(f'swarm{car_id}')
        gazebostate_car.append((t,msg.pose[pose_id]))

def read_one_car(bag_dir,car_id,check_image=False):
    bag = rosbag.Bag(bag_dir)
    topic_info = bag.get_type_and_topic_info()
    print(topic_info)
    print(bag_dir)

    img_0 = []
    img_1 = []
    img_2 = []
    img_3 = []
    img_time = []

    ## read images
    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[ '/cam0/image_raw',
                                                    '/cam1/image_raw',
                                                    '/cam2/image_raw',
                                                    '/cam3/image_raw'] ):
        # print(f"Received message on topic {topic} at time {t}")
        # print(f"Received message on topic {topic} at time {msg.header}")
        #pdb.set_trace()
        cnt = cnt + 1
        if cnt < 200:
            continue
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        if 'cam0' in topic:
            img_0.append(cv_image)
            #print(f"Received message on topic {topic} at time {t}")
            #print("image time diff:",convert_time(t - msg.header.stamp)/1e06)
            img_time.append(t)  ## use camera 0 as baseline
        elif 'cam1' in topic:
            img_1.append(cv_image)
        elif 'cam2' in topic:
            img_2.append(cv_image)
        elif 'cam3' in topic:
            img_3.append(cv_image)
    
    #total_img = topic_info.topics['/cam0/image_raw'].message_count
    ## read odom
    assert (len(img_0) == len(img_1), "total_img == len(img_0)") and \
           (len(img_1) == len(img_2), "total_img == len(img_1)") and \
           (len(img_2) == len(img_3), "total_img == len(img_2)") 

    total_img = np.min([len(img_0),len(img_1),len(img_2),len(img_3)])
    print(f"total_img {total_img}")
    
    def Reverse(lst):
        return [ele for ele in reversed(lst)]
    
    car = Car()
    car.host_id = car_id
    odom_car_0 = []
    gazebostate_car_0 = []
    
    ######## use robot odom
    for topic, msg, t in bag.read_messages(topics=['/robot/odom']):
        #print(f"Received message on topic {topic} at time {t}")
        #print("odom time diff:",convert_time(t - msg.header.stamp)/1e06)
        #print(msg.header.stamp)
        msg_offset = msg
        msg_offset.header.stamp = t
        odom_car_0.append(msg_offset)
        car.add_odom(msg_offset)
   # pdb.set_trace()
    ######## use gazebo state
    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
        #print(t)
        pose_id = msg.name.index(f'swarm{car_id}')
        gazebostate_car_0.append((t,msg.pose[pose_id]))
        car.add_gazebo_state((t,msg.pose[pose_id]))

    ic = ImageConcatenater()
    cutted_img_height = ic.get_cutted_img_height()
    start_id,_ = ic.compute_crop_size()
    original_img_width = ic.get_original_img_width()

    for i in range(total_img):
        img0 = img_0[i][-cutted_img_height:, start_id:original_img_width-start_id : ]
        img1 = img_1[i][-cutted_img_height:, start_id:original_img_width-start_id : ]
        img2 = img_2[i][-cutted_img_height:, start_id:original_img_width-start_id : ]
        img3 = img_3[i][-cutted_img_height:, start_id:original_img_width-start_id : ]

        img_list = [img0,img1,img2,img3]
        result = cv2.hconcat((img_list))
        car.add_image(result)
        car.add_image_time(img_time[i])
        car.add_raw_image(img_list)
        ####### Check single agent time stamps ###########
        odom = get_odom(odom_car_0,img_time[i])
        gazebo_state = get_gazebo_state(gazebostate_car_0,img_time[i])
       # pdb.set_trace()
        #print("odom and image time diff (ms):",convert_time(img_time[i] - odom.header.stamp)/ 1e06)
        #print("gazebo state and image time diff (ms):",convert_time(img_time[i] - gazebo_state[0])/ 1e06)
        
        ##################################################
        if check_image:
            cv2.imshow("omni_image",result)
            ##plt.matshow(result)
            #plt.show()
            cv2.waitKey()
    return car

def read_real_world_bag(bag_dir,car_id,check_image=False):
    bag = rosbag.Bag(bag_dir)
    topic_info = bag.get_type_and_topic_info()
    print(topic_info)

    img_0 = []
    img_1 = []
    img_2 = []
    img_3 = []
    img_time = []

    ## read images
    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[ '/cam0/image_raw',
                                                    '/cam1/image_raw',
                                                    '/cam2/image_raw',
                                                    '/cam3/image_raw'] ):
        #print(f"Received message on topic {topic} at time {t}")
        #print(f"Received message on topic {topic} at time {msg.header}")
        
        cnt = cnt + 1
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        if 'cam0' in topic:
            img_0.append(cv_image)
            #print(f"Received message on topic {topic} at time {t}")
           # print(f"image time diff:{convert_time(t - msg.header.stamp)/1e06} ms")
            img_time.append(msg.header.stamp)  ## use camera 0 as baseline
        elif 'cam1' in topic:
            img_1.append(cv_image)
        elif 'cam2' in topic:
            img_2.append(cv_image)
        elif 'cam3' in topic:
            img_3.append(cv_image)
    #total_img = topic_info.topics['/cam0/image_raw'].message_count
    ## read odom
    assert (len(img_0) == len(img_1), "total_img == len(img_0)") and \
           (len(img_1) == len(img_2), "total_img == len(img_1)") and \
           (len(img_2) == len(img_3), "total_img == len(img_2)") 

    total_img = np.min([len(img_0),len(img_1),len(img_2),len(img_3)])
    print(f"total_img {total_img}")
   # pdb.set_trace()
    
    def Reverse(lst):
        return [ele for ele in reversed(lst)]
    
    car = Car()
    car.host_id = car_id
    odom_car_0 = []
    print(f"total_img {total_img}")

    ######## use vicon  ################
    for topic, msg, t in bag.read_messages(topics=[f'/vicon/VSWARM{car_id}/VSWARM{car_id}']):
        odom_car_0.append(msg)
        car.add_odom(msg)

    ########################
    ## image undistortion ##
    ########################

    ic = ImageConcatenater()
    cutted_img_height = ic.get_cutted_img_height()
    start_id,_ = ic.compute_crop_size()
    original_img_width = ic.get_original_img_width()

    cam_para = Car_param(car_id,'cali_file')
    for i in range(total_img):
        map1,map2 = get_map_from_intrinsics(0,cam_para)
        img_0[i] = cv2.remap(img_0[i], map1, map2, cv2.INTER_LINEAR)
        map1,map2 = get_map_from_intrinsics(1,cam_para)
        img_1[i] = cv2.remap(img_1[i], map1, map2, cv2.INTER_LINEAR)
        map1,map2 = get_map_from_intrinsics(2,cam_para)
        img_2[i] = cv2.remap(img_2[i], map1, map2, cv2.INTER_LINEAR)
        map1,map2 = get_map_from_intrinsics(3,cam_para)
        img_3[i] = cv2.remap(img_3[i], map1, map2, cv2.INTER_LINEAR)

        img0 = img_0[i][90:img_0[i].shape[0]-90, : ]
        img1 = img_1[i][90:img_0[i].shape[0]-90, : ]
        img2 = img_2[i][90:img_0[i].shape[0]-90, : ]
        img3 = img_3[i][90:img_0[i].shape[0]-90, : ]
        #print(img0.shape)
        img_list = [img0,img1,img2,img3]
        result = cv2.hconcat((img_list))
        car.add_image(result)
        car.add_image_time(img_time[i])
        car.add_raw_image(img_list)
        ####### Check single agent time stamps ###########
        #odom = get_odom(odom_car_0,img_time[i])
        #pdb.set_trace()
        #print("odom and image time diff (ms):",convert_time(img_time[i] - odom.header.stamp)/ 1e06)
        
        ##################################################
        if check_image:
            cv2.imshow("omni_image",result)
            cv2.waitKey(1)
    return car

def convert_pose_msg_to_mat(pose):
    translation = [pose.position.x, pose.position.y, pose.position.z]
    rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z ,pose.orientation.w]
    rotation_ = R.from_quat(rotation)
    translation_matrix = tf.transformations.translation_matrix(translation)
    translation_matrix[0:3,0:3] = rotation_.as_matrix()
    return translation_matrix

def convert_Transform_to_mat(pose):
    translation = [pose.translation.x, pose.translation.y, pose.translation.z]
    rotation = [pose.rotation.x, pose.rotation.y, pose.rotation.z ,pose.rotation.w]
    rotation_ = R.from_quat(rotation)
    translation_matrix = tf.transformations.translation_matrix(translation)
    translation_matrix[0:3,0:3] = rotation_.as_matrix()
    return translation_matrix


def project_onto_image(p_in_camera):
    ic = ImageConcatenater()
    focal_len = ic.get_cropped_focal()
    cx = ic.get_cropped_cx()
    cy = ic.get_cropped_cy()
    #pdb.set_trace()
    u = focal_len * p_in_camera[0] / p_in_camera[2] + cx
    v = focal_len * p_in_camera[1] / p_in_camera[2] + cy
    return (u,v)

def project_onto_image2(p_in_camera,cam_param):
    cam_K = cam_param.get_undistort_K()
    focal_len = cam_K[0][0]
    cx = cam_K[0][2]
    cy = cam_K[1][2]
    #print("focal_len,cx,cy:",focal_len,cx,cy)
    #pdb.set_trace()
    u = focal_len * p_in_camera[0] / p_in_camera[2] + cx
    v = focal_len * p_in_camera[1] / p_in_camera[2] + cy
    return (u,v)

def project_cam(img_h,single_img_w,p_in_body,cam_param):
    camera_body_3 = get_cam_body_extrinsics(3,cam_param)
    p_in_camera_3 = camera_body_3 @ p_in_body

    camera_body_2 = get_cam_body_extrinsics(2,cam_param)
    p_in_camera_2 = camera_body_2 @ p_in_body

    camera_body_1 =  get_cam_body_extrinsics(1,cam_param)
    p_in_camera_1 = camera_body_1 @ p_in_body

    camera_body_0 = get_cam_body_extrinsics(0,cam_param)
    p_in_camera_0 = camera_body_0 @ p_in_body

    p_candidates = [p_in_camera_0,p_in_camera_1,p_in_camera_2,p_in_camera_3]
    res = []
    for id,p in enumerate(p_candidates):
        if p[2] < 0:
            continue
        u,v = project_onto_image2(p,cam_param)
        #print((u,v,id))
        if u >= 0 and u < single_img_w and v >=0 and v < img_h:
            u = u + id * single_img_w
            res.append((u,v,id))
    if len(res) != 1 and len(res) != 0:
        #print("len(res):",len(res))
        x = 0
        y = 0
        for k in range(len(res)):
            x = x + res[k][0]
            y = y + res[k][1]
        x = x / len(res)
        y = y / len(res)
        
        #pdb.set_trace()
        res = [(x,y,res[0][2])]
    #assert len(res) == 1 or len(res) == 0
    return res

def generate_key_point_labels(img_h,img_w,relative_pose):
    """
    generate semantic key point labels for ONE target
    """
    pass

def generate_v_attention_labels(img_h,img_w,relative_pose,cam_param):
    """
    generate visual attention labels for ONE target
    """
    p_in_body =  relative_pose @ np.array([0,0,0,1]).T
    single_img_wid = ImageConcatenater().compute_cropped_width()
    p = project_cam(img_h,single_img_wid,p_in_body,cam_param)
    p_np = np.array(p)
    body_depth = np.linalg.norm(p_in_body[0:2])
    #print(p_np)
    hm = generate_hm(img_h,img_w,p_np,s = int(20 / body_depth))
    return hm


ic = ImageConcatenater()
def generate_motion_compensation_img(img_current,img_previsous,relative_pose,camera_body,show=False):
    """
    generate motion compensated image.[img_h,img_w,2] -> (gray,motion_flow)
    relative_pose: relative pose of body frame
    camera_body: extrinsics between camera and body
    """
    ex = camera_body @ relative_pose @ np.linalg.inv(camera_body)
    focal_len = ic.get_cropped_focal()
    cx = ic.get_cropped_cx()
    cy = ic.get_cropped_cy()
    K_mat = np.array([[focal_len,     0,      cx], 
                      [0,        focal_len,   cy],
                      [0,             0,       1]])

    H = K_mat @ ex[0:3,0:3] @ np.linalg.inv(K_mat)  ## current -> last
    w = img_current.shape[1]
    h = img_current.shape[0]
    warped_current = cv2.warpPerspective(img_current, H, (w, h))
    
    #### compute optical flow ##########################
    hsv = np.zeros_like(img_previsous)
    if warped_current.shape[2] == 3:
        warped_current_gray = cv2.cvtColor(warped_current, cv2.COLOR_BGR2GRAY)
    if img_current.shape[2] == 3:
        current_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    if img_previsous.shape[2] == 3:
        pre_gray = cv2.cvtColor(img_previsous, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(pre_gray,warped_current_gray, None, **fb_params)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    motion_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #motion_flow = cv2.copyTo(motion_flow,bbox_mask)
    motion_flow = cv2.cvtColor(motion_flow, cv2.COLOR_BGR2GRAY)
    #motion_flow = np.expand_dims(motion_flow, axis=2)

    if show:
        overlapping1 = cv2.addWeighted(img_current, 0.5, img_previsous, 0.5, 0)
        overlapping2 = cv2.addWeighted(warped_current, 0.5, img_previsous, 0.5, 0)
        overlap_list = [overlapping1,overlapping2]
        result = cv2.hconcat((overlap_list))
        cv2.imshow("result",result)
        cv2.imshow("motion_flow",motion_flow)
        cv2.imshow("current_gray",current_gray)
        cv2.waitKey(1)
        
    return current_gray, motion_flow

def is_homogeneous(lst):
    if not lst:
        return True
    
    first_shape = np.shape(lst[0])
    for item in lst:
        if np.shape(item) != first_shape:
            return False
    return True

def generate_bbox_labels(img_h,img_w,relative_pose,cam_param):
    """
    generate bounding box labels for ONE target
    """
    robot_radius = 0.15
    points_on_car = []
    for theta in np.linspace(0,2*np.pi,20):
        p_on_car_circle = [robot_radius * np.cos(theta),
                           robot_radius * np.sin(theta),
                           0.0,  1 ]
        points_on_car.append(p_on_car_circle)
    points_on_car.append([0,       0,      0.1,           1])
    points_on_car = np.array(points_on_car)

    p_in_body_all =  relative_pose @ points_on_car.T
    bbox_pro_pts = []
    single_img_wid = ImageConcatenater().compute_cropped_width()
    
    for i in range(p_in_body_all.shape[1]):
        p_ = project_cam(img_h,single_img_wid,p_in_body_all[:,i],cam_param)
        #print()
        if len(p_) > 0:
            bbox_pro_pts.append(p_)
    print(np.array(bbox_pro_pts).astype(np.float32))
    #pdb.set_trace()
    p_np = None
    
    if len(bbox_pro_pts) > 0 and is_homogeneous(bbox_pro_pts):
        p_np = np.array(bbox_pro_pts).squeeze(1)
    return p_np

def generate_bbox_mask(pts_list_np, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for pts in pts_list_np:
        x_min, y_min, x_max, y_max = pts[1], pts[2], pts[3], pts[4]
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    return mask

fb_params = dict(pyr_scale = 0.5,
                 levels = 3,
                 winsize = 10,
                 iterations = 1,
                 poly_n = 5,
                 poly_sigma = 3,
                 flags = 0)

def check_and_save_img(img,data_path,img_cnt, mono=True, show=False):
    """
    check and save image
    """
    if mono:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{data_path}/{img_cnt:05d}.png', img)
    if show:
        cv2.imshow('gray image',img)
        cv2.waitKey()


def check_and_save_img_motion(pts_list_np,cated_img,data_path,img_cnt,show=False):
    """
    check and save image, save optical flow
    """
    ################# gernerate mask for motion flow ################
    bbox_mask = generate_bbox_mask(pts_list_np,cated_img.shape)
    # bbox_mask_flip = cv2.flip(bbox_mask,-1)

    #################################################################
    motion_flow = cv2.copyTo(cated_img[:,:,1],bbox_mask)
    cated_img[:,:,1] = copy.deepcopy(motion_flow)
    #print(data_path,cated_img.shape)
    np.save(f'{data_path}/{img_cnt:05d}.npy',cated_img)
    # temp = np.load(f'{data_path}/{img_cnt:05d}.npy')
    # print("save shape:",temp.shape)

    if show:
        cv2.imshow('gray image',cated_img[:,:,0])
        cv2.imshow('motion image masked',cated_img[:,:,1])
        cv2.waitKey()

def check_and_save_bbox(img,pts_vec,data_path,img_cnt,show=False):
    """
    check and save bbox label
    """
    if len(pts_vec) == 0:
        print(f"no target in this image {img_cnt}")
        return
    
    import copy
    img_check = copy.deepcopy(img)
    width = img.shape[1]
    height = img.shape[0]
    
    pts_list = []
    pts_list_scale = []
    
    for id,pts_on_one_car in enumerate(pts_vec):
        same_cam = np.all(pts_on_one_car[:, 2] == pts_on_one_car[0, 2])
        if same_cam == False and \
        (np.all(np.logical_or(pts_on_one_car[:, 2] == 3, pts_on_one_car[:, 2] == 0))):
            continue
       # print(pts_on_one_car[:, 2])
       # print("same_cam:",same_cam)
        points = pts_on_one_car[:,0:2].astype(np.int64)
        #print("pts_on_one_car\n",pts_on_one_car)
        offset = 0
        p_x_max = np.max(points[:,0]) + offset
        p_x_min = np.min(points[:,0]) - offset
        p_y_max = np.max(points[:,1]) + offset
        p_y_min = np.min(points[:,1]) - offset
        ####  original format ####
        pts_list.append([id,p_x_min,p_y_min,p_x_max,p_y_max])
        ####  yolo format ####
        pts_list_scale.append([0,
                         (p_x_min + p_x_max) / 2 / width,   ## x_center
                         (p_y_min + p_y_max) / 2 / height,  ## y_center
                         (p_x_max - p_x_min ) / width,
                         (p_y_max - p_y_min ) / height]) 

        if show:
            cv2.rectangle(img_check,(p_x_min,p_y_min),(p_x_max,p_y_max),(0,255,0))
    pts_list_np = np.array(pts_list)
    pts_list_scale_np = np.array(pts_list_scale)
    #np.save(f'{data_path}/{img_cnt:05d}.npy',pts_list_np)
    #temp = np.load(f'{data_path}/{img_cnt:05d}.npy')
    if len(pts_list_scale) == 0:
        return
    fmt = ['%d'] + ['%.6f'] * (pts_list_scale_np.shape[1] - 1)
    np.savetxt(f'{data_path}/{img_cnt:05d}.txt', pts_list_scale_np, fmt=fmt) 
    #pdb.set_trace()

    if show:
        img_flip = cv2.flip(img_check,-1)
        cv2.imshow("omni_image_bbx_check",img_flip)
        cv2.waitKey()
    return pts_list_np

def check_and_save_vattention(img,hm_total_v_att,v_att_label_path,img_cnt,show=False):
    """
    check and save visual attention data
    """
    print("hm_total shape:",hm_total_v_att.shape)
    valid_target_num = hm_total_v_att.shape[2]
    print("valid_target_num:",valid_target_num)
    if show:
        hm_total_v_att = ((255* hm_total_v_att)).astype(dtype=np.uint8)
        if valid_target_num != 0:
            heat_map_total = np.array(hm_total_v_att)
            heat_map_total = cv2.cvtColor(heat_map_total, cv2.COLOR_BGR2RGB)
        else:
            heat_map_total = np.zeros((hm_total_v_att.shape[0],hm_total_v_att.shape[1],1),dtype=np.uint8)
            
        heat_map_trans = cv2.applyColorMap(heat_map_total, cv2.COLORMAP_JET)
        overlapping = cv2.addWeighted(img, 0.5, heat_map_trans, 0.5, 0)
        overlapping_flip = cv2.flip(overlapping,-1)
        cv2.imshow("overlap",overlapping_flip)
        cv2.waitKey()
        
    #hm_total_v_att = cv2.flip(hm_total_v_att,-1)
    print(f"v_att_label_path {v_att_label_path}")
    np.save(f'{v_att_label_path}/{img_cnt:05d}.npy',hm_total_v_att)
    # temp = np.load(f'{v_att_label_path}/{img_cnt:05d}.npy')
    # print(temp.shape)

def preprocess(img,scale):
    """
    preprocess image for v attention inference
    """
    img = cv2.resize(img,dsize=None,fx=scale,fy=scale,interpolation = cv2.INTER_NEAREST)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # if (img > 1).any():
    #     img = img / 255.0
    img = img / 255
    return img

def convert_np(img_tensor):
    """
    convert v attention prediction tensor to numpy array
    """
    return img_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) / (max_val - min_val)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    return normalized_image

def normalize_array(arr):
    if len(arr) == 0:
        return arr
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = ((np.array(arr) - min_val) / (max_val - min_val)) * 1
    return normalized_arr
