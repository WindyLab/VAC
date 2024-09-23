import numpy as np
import matplotlib.pyplot as plt
import cv2
from data_utils import *
from path_utils import *
import copy
from cfg_vatt import config_train as cfg

def main(car_list,graph_edges,sim_id,data_prefix):
    leader_host_id = 0
    ic = ImageConcatenater()
    img_h = ic.get_cutted_img_height()
    img_w = ic.compute_cropped_width() * 4
    #print(img_h,img_w)
    #pdb.set_trace()
    image_cnt = 0
    data_path = join_and_create(f'{data_prefix}_{sim_id}',"data")
    bbx_label_path = join_and_create(f"{data_prefix}_{sim_id}", "labels")
    vatt_label_path = join_and_create(f"{data_prefix}_{sim_id}", "att_labels")
    pose_last_rf = None
    pre_frame_list = []
    for car_cur in car_list:
        if car_cur.host_id == leader_host_id:
            continue
        cam_para = Car_param(999,'cali_file')
        for img_id,img in enumerate(car_cur.image_vec):
            t_target = car_cur.image_time_vec[img_id]
            raw_img_list = car_cur.raw_img_list_list[img_id]
            odom_current = get_odom(car_cur.odom_vec,t_target)
            gazebo_state_current = get_gazebo_state(car_cur.gazebo_state,t_target)

            ###############  Prepare data container for labels  ########
            hm_total_v_att = np.zeros((img_h, img_w, 1), dtype = np.float32)
            pts_vec = []

            valid_target_num = 0
            for car_target in car_list:

                #odom_target = get_odom(car_target.odom_vec,t_target )
                gazebo_state_target = get_gazebo_state(car_target.gazebo_state,t_target)
                
                # print("Agent odom and odom time diff (ms):",convert_time(odom_current.header.stamp - odom_target.header.stamp) / 1e06)
                #print("Agent img and odom time diff (ms):",convert_time(t_target - odom_target.header.stamp) / 1e06)
                #pdb.set_trace()
                #print("Agent odom and odom time diff (ms):",convert_time(gazebo_state_current[0] - gazebo_state_target[0]) / 1e06)
                print("Agent img and odom time diff (ms):",convert_time(t_target - gazebo_state_target[0]) / 1e06)
                pose_current = gazebo_state_current[1]
                pose_target = gazebo_state_target[1]

                # pose_current = odom_current.pose.pose
                # pose_target = odom_target.pose.pose
                pose_cu_tf = convert_pose_msg_to_mat(pose_current)
                pose_tar_tf = convert_pose_msg_to_mat(pose_target)
                delta = np.linalg.inv(pose_cu_tf) @ pose_tar_tf

                #print(pose_current,'\n',pose_target)
                #print(delta)
                print(f"current car {car_cur.host_id} |target car {car_target.host_id}")
                #print("interaction graph:\n",graph_edges)

                ### 1. generate bbox label ###
                pts = generate_bbox_labels(img_h,img_w,delta,cam_para)
                if pts is not None:
                    pts_vec.append(pts)
                
                ### 2. generate v attention label ###
                ## Generate visual attention label ONLY for valid edges
                if graph_edges[car_cur.host_id,car_target.host_id] != 0:
                    print(f"with connection!{car_cur.host_id} {car_target.host_id}")
                    hm_vatten = generate_v_attention_labels(img_h,img_w,delta,cam_para)
                    if hm_vatten.shape[2] != 0:
                        hm_total_v_att = hm_total_v_att + hm_vatten
                    print('\n')
                    
            # ### 3. gerneate motion compensated image ###
            if img_id > 0 and cfg['with_motion']:
                delta_current_body = np.linalg.inv(pose_last_rf) @ pose_cu_tf #convert current frame to last frame
                img_cat = []
                for cam_id in range(4):
                    cam_body = get_cam_body_extrinsics(cam_id,cam_para)
                    current_gray, motion_flow = generate_motion_compensation_img(raw_img_list[cam_id],pre_frame_list[cam_id], \
                                                    delta_current_body,cam_body,show=False)
                    motion_flow = np.expand_dims(motion_flow, axis=2)
                    current_gray = np.expand_dims(current_gray, axis=2)
                    img_save = np.concatenate((current_gray,motion_flow),axis=2)
                    img_cat.append(img_save)
                    print(f"{current_gray.shape} {motion_flow.shape}")
                ## 
                cat_motion = cv2.hconcat(img_cat)
                cv2.imshow("cated",cat_motion[:,:,0])
                cv2.imshow("cated motion",cat_motion[:,:,1])
                cv2.waitKey(1)

            pre_frame_list = []
            for cam_id in range(4):
                pre_frame_list.append(copy.deepcopy(raw_img_list[cam_id]))
            assert hm_total_v_att.shape[2] != 0
            
            if img_id > 0:
                check_and_save_vattention(img,hm_total_v_att,vatt_label_path,image_cnt,show=False)
                pts_list_np = check_and_save_bbox(img,pts_vec,bbx_label_path,image_cnt,show=True)
                if cfg['with_motion']:
                    check_and_save_img_motion(pts_list_np,img,cat_motion,data_path,image_cnt,show=False)
                else:
                    check_and_save_img(img,data_path,image_cnt,mono=False,show=False)

            pose_last_rf = copy.deepcopy(pose_cu_tf)
            image_cnt = image_cnt + 1


if __name__ == '__main__':
    bag_path = './'  ## bag path
    graph_path = './graph.npy'  ### interaction graph
    agent_num = 5
    host_id = list(range(agent_num))
    sim_id = 0
    check_img = False
    graph_edges = np.load(graph_path)
    print(f'graph_edges shape {graph_edges}')
    car_list = []
    if cfg['with_motion']:
        data_prefix = 'gazebo_motion_data'
    else:
        data_prefix = 'gazebo_data'
    for i in host_id:
        bag_dir = f'{bag_path}_{sim_id}_swarm{i}.bag'
        if os.path.exists(bag_dir):
            car_list.append(read_one_car(bag_dir,i,check_img))
        else:
            print(f'DATA: {bag_dir} Not exist!!!')
    main(car_list,graph_edges,sim_id,data_prefix)