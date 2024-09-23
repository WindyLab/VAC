import numpy as np
import json,cv2
import pdb

class Car_param():
    def __init__(self,host_id,file_path):
        super(Car_param, self).__init__()
        self.host_id = host_id
        self.cam_K = dict()
        self.cam_dist = dict()
        self.cam_ex = dict()
        f = open(f'{file_path}/{host_id:03d}.json', 'r')
        para = f.read()
        self.cam_para = json.loads(para)

    def get_cam_K(self,cam_id):
        return self.cam_para[f'cam{cam_id}_K']

    def get_cam_d(self,cam_id):
        return self.cam_para[f'cam{cam_id}_d']
    
    def get_cam_body(self):
        return self.cam_para['cam_body']
    
    def get_undistort_K(self):
        cam_new_K = np.array(    [  [205,  0,        205 ],
                                [0,    205,      180],
                                [0,      0,        1]])
        return cam_new_K
    
    def get_undistort_sz(self):
        return (410,360)
    
def get_map_from_intrinsics(cam_id,cam_Param):
    cam_d = cam_Param.get_cam_d(cam_id)
    cam_K = cam_Param.get_cam_K(cam_id)
    cam_K = np.array(cam_K)
    cam_d = np.array(cam_d)
    sz = cam_Param.get_undistort_sz()
    cam_new_K = np.array( cam_Param.get_undistort_K() )
    map1,map2 = cv2.initUndistortRectifyMap(cam_K,cam_d,None, cam_new_K,sz,cv2.CV_32FC1)
    return map1,map2

def get_x_mat(theta):
    transformation_matrix = np.eye(4,4)
    transformation_matrix[1,1] = np.cos(theta)
    transformation_matrix[1,2] = -np.sin(theta)
    transformation_matrix[2,1] = np.sin(theta)
    transformation_matrix[2,2] = np.cos(theta)
    return transformation_matrix

def get_z_mat(theta):
    transformation_matrix = np.eye(4,4)
    transformation_matrix[0,0] = np.cos(theta)
    transformation_matrix[0,1] = -np.sin(theta)
    transformation_matrix[1,0] = np.sin(theta)
    transformation_matrix[1,1] = np.cos(theta)
    return transformation_matrix

def get_cam_body_extrinsics(cam_id,cam_Param):
    cam_body = cam_Param.get_cam_body()
    camera_body_3 = get_z_mat(np.pi) @ get_x_mat(np.pi/2 + cam_body[0][1]) @ get_z_mat(0 + cam_body[0][2])
    camera_body_2 = get_z_mat(np.pi) @ get_x_mat(np.pi/2+ cam_body[0][1]) @ get_z_mat(np.pi/2 + cam_body[0][2])
    camera_body_1 =  get_z_mat(np.pi) @ get_x_mat(np.pi/2+ cam_body[0][1]) @ get_z_mat(np.pi + cam_body[0][2])
    camera_body_0 = get_z_mat(np.pi) @ get_x_mat(np.pi/2+ cam_body[0][1]) @ get_z_mat(-np.pi/2 + cam_body[0][2])
    temp = [camera_body_0,camera_body_1,camera_body_2,camera_body_3]
    return temp[cam_id]