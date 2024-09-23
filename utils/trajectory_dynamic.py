import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
from common_utils import *
import torch
from formation_utils import *
from matplotlib.patches import Rectangle
import argparse
from tqdm import tqdm
import os
import shutil

def animate(t,pos_vec,vel_vec,edge_vec,ax,rectangle_boundary,rectangle_unrelevant,N_agent_total,
            robot_radius,
            perception_radius):
    p_all = pos_vec[t]
    v_all = vel_vec[t]
    edge_c = edge_vec[t]
    ax.cla()
    ax.add_patch(rectangle_boundary)
    ax.add_patch(rectangle_unrelevant)
    draw_robots(ax,p_all,v_all,N_agent_total,robot_radius,edge_c)
    #draw_perception_range(ax,p_all,N_agent_total,perception_radius)
    plt.pause(0.00001)
    plt.ioff()

def trajec_dynamic_test(args):
    ############  Parameters  ##################
    N_agent = [3,2]
    leader_id = [0,0]
    N_agent_total = np.sum(N_agent)
    total_leader_id = []
    for i in range(len(N_agent)):
        id = np.ones(N_agent[i]) * leader_id[i]
        if i == 0:
            total_leader_id = id
        else:
            total_leader_id = np.concatenate((total_leader_id,id),axis=0)

    interactive_show = args.intershow
    save_ani = args.save_ani
    perception_radius = args.perception_radius
    mode = args.motion_mode
    robot_radius = 0.15
    seq_len = args.obs_frames + args.rollouts
    obs_frames = args.obs_frames
    rollouts = args.rollouts
    start_v = torch.rand(N_agent_total,2) * 1
    start_theta = torch.zeros(N_agent_total,)

    disturb_id = []
    length_out = 3
    x_min = -length_out
    x_max = length_out
    y_min = -length_out
    y_max = length_out
    
    length_center = 1.4
    random_target_ratio = 0.95
    x_min_center = -length_center
    x_max_center = length_center
    y_min_center = -length_center
    y_max_center = length_center

    start_point_vec = []
    start_point_vec.append(generate_main_initial_pos(N_agent[0],robot_radius,-1,1,-1,1,R=2))

    for i in range(1,len(N_agent)):
        start_point_vec.append(generate_initial_pos(     N_agent[i],
                                                        robot_radius,
                                                        x_max_center,
                                                        x_min_center,
                                                        y_max_center,
                                                        y_min_center) )
    ################################################################

    start_point_vec = torch.cat(start_point_vec,dim=0)
    edge = None
    for i,n in enumerate(N_agent):
        edge_c  = generate_random_graph(n,mode=2)
        if i == 0:
            edge = edge_c
        else:
            edge = combine_edge(edge,edge_c)
            
    #print(edge)
    edge_input = copy.deepcopy(edge)
    v_all = start_v
    p_all = start_point_vec
    theta_all = start_theta

    
    ############### check start point ########
    # for ag in range(N_agent_main):
    #     draw_circle = plt.Circle((start_pt[ag][0], start_pt[ag][1]), robot_radius,fill=False,color='blue')
    #     text_obj = plt.text(start_pt[ag][0],start_pt[ag][1],str(ag))
    #     ax.add_artist(draw_circle)
    # for ag in range(N_agent_unreleve):
    #     draw_circle = plt.Circle((start_pt_unrelevant[ag][0], start_pt_unrelevant[ag][1]), robot_radius,fill=False,color='blue')
    #     text_obj = plt.text(start_pt_unrelevant[ag][0],start_pt_unrelevant[ag][1],str(ag))
    #     ax.add_artist(draw_circle)
    # plt.show()
    # ###################

    ################  check leader trajectory ################# 
    # plt.subplots()
    # pt = start_pt[0]
    # print(pt)
    # for t in range(100):
    #     pt = pt + torch.tensor(compute_circle_leader_v(t))
    #     plt.scatter(pt[0],pt[1])
    # plt.show()
    ###########################################################

    # mig_target = start_pt + torch.tensor((10,0))

    # #print("edge_mat_input:\n",edge_mat_input)
    # #print("edge_mat_input:\n",edge_mat_unrele)

    pos_vec = []
    vel_vec = []
    theta_vec = []
    edge_vec = []
    random_walk_target = (torch.rand((2,)) - 0.5)  * (length_center * random_target_ratio)
    #print(random_walk_target)

    for t in range(seq_len):
        ##########  Main part  ##############
        if t > 0:
            v_all = torch.zeros_like(v_all)
            theta_all = torch.zeros_like(theta_all)
        relative_pos = (-1) * (p_all.unsqueeze(1) - p_all.unsqueeze(0))
        relative_pos = relative_pos + torch.rand(relative_pos.shape) * 0.00001 ## Add noise
        relative_dist = torch.norm(relative_pos,dim=2) + 100 * torch.eye(N_agent_total)
        #temp = torch.norm(relative_pos,dim=2)
        temp_edge = relative_dist < perception_radius
        if args.collision_warning and (relative_dist < 2 * robot_radius ).any():
           print("warning!!collision outside!! frame:",t)
           print(relative_dist)
           print('\n')

        #############################################
        for i in range(N_agent_total):
            id = total_leader_id[i].astype(np.int8)
            mig_target = p_all[id] - p_all.unsqueeze(0)
            #pdb.set_trace()
            if t > 0:
                v_last_x = vel_vec[t-1][i][0]
                v_last_y = vel_vec[t-1][i][1]
            else:
                v_last_x = start_v[i][0]
                v_last_y = start_v[i][1]
            if i in disturb_id:
                continue
            if i == leader_id[0]: ### leader
                #v = torch.tensor(compute_circle_leader_v(t))
                if mode == 0:
                    v_norm = 4 / seq_len
                    v = torch.tensor(compute_retangular_leader_v(t,seq_len,v_norm))
                elif mode == 1:
                    v = torch.tensor(compute_circle_leader_v(t))
            # elif i == leader_id[1]:  ## leader
            #     v = compute_v_PD (                   random_walk_target[0],
            #                                             random_walk_target[1],
            #                                             p_all[i][0],
            #                                             p_all[i][1],
            #                                             v_all[i][0],
            #                                             v_all[i][1],
            #                                             v_last_x,
            #                                             v_last_y )
            #     v = torch.tensor(v)
            #   #  print("===============leader_id[1]",v,random_walk_target,v_last_x)
            #   #  print(f'{v_last_x} {v_last_y} {random_walk_target} {v}')
            else:
                v = compute_indivisual_v_v2(i,relative_pos,mig_target.squeeze(0)[i],temp_edge,torch.tensor([v_last_x,v_last_y]),
                                           k_sep = 0.7,k_mig= 0.01,k=0.01)
                #print(f'temp_edge {temp_edge}')
              # v = compute_indivisual_v(i,relative_pos,mig_target.squeeze(0)[i],temp_edge)
                v = torch.tensor(v)
                #print(v)

            v_all[i] = v
            raw_az = torch.atan2(v[1],v[0]).item()
            theta_all[i] = raw_az
        if torch.norm(p_all[leader_id[1]] - random_walk_target) < 0.03:
            random_walk_target = (torch.rand((2,)) - 0.5) * (length_center * random_target_ratio)
        p_all = p_all + v_all
        pos_vec.append(p_all.numpy())
        vel_vec.append(v_all.numpy())
        theta_vec.append(theta_all.numpy())
        edge_vec.append(temp_edge.numpy())
        
    p_all_save = np.concatenate((pos_vec,vel_vec),                              axis=2)
    final_pos = p_all_save
    current_data  =  final_pos[0:obs_frames,  :,  0:args.feat_dim]
    current_label =  final_pos[0:obs_frames,  :,  0:args.feat_dim]
    current_label = torch.tensor(current_label)
    edge = edge.unsqueeze(0).repeat(rollouts,1,1)
    history_edge_label = np.array(edge_vec).astype(np.int8)[0:obs_frames,:,:]
    current_label = torch.cat((current_label,torch.tensor(history_edge_label)),dim=2).numpy()
   
    if not hasattr(trajec_dynamic_test, "file_id"):
        trajec_dynamic_test.file_id = 0
    plt.close()
    if interactive_show:
        plt.ion()
        fig,ax = plt.subplots(figsize=(10,10))
        rectangle_boundary = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='b', facecolor='none')
        rectangle_unrelevant = Rectangle((x_min_center, y_min_center), x_max_center - x_min_center, y_max_center - y_min_center, linewidth=1, edgecolor='r', facecolor='none')
        ax.set_aspect(1)
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        ani = animation.FuncAnimation(fig, animate, frames=seq_len, fargs=(pos_vec,
                                    vel_vec,edge_vec,ax,rectangle_boundary,rectangle_unrelevant,N_agent_total,
            robot_radius,perception_radius), interval=100)
        if True:
            ani.save(f'animation{trajec_dynamic_test.file_id}.gif', writer='pillow', fps=10)
            trajec_dynamic_test.file_id += 1
        plt.show()
        

    return current_data, current_label,N_agent_total

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset_size', type=int, default=70000)
    parser.add_argument('--randomseed', type=int, default=102)
    parser.add_argument('--obs_frames', type=int, default=100)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--feat_dim', type=int, default=4)
    parser.add_argument('--motion_mode', type=int, default=1)  ## 0 : retan 1: circular
    parser.add_argument('--perception_radius', type=float, default=1.6)
    parser.add_argument('--intershow', type=bool, default=False)
    parser.add_argument('--save_ani', type=bool, default=False)
    parser.add_argument('--collision_warning', type=bool, default=False)
    
    args = parser.parse_args()
    obs_frames = args.obs_frames
    rollouts = args.rollouts
    seq_length = rollouts + obs_frames
    feat_dim = 2
    torch.manual_seed(args.randomseed)
    num_instances = args.dataset_size
    target_path = './'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    all_data = []
    all_labels = []
    all_edge_labels = []
    N_agent = 0

    for test_case in tqdm(range(num_instances)):
        data,label,N_agent = trajec_dynamic_test(args)
        all_data.append(data)
        all_labels.append(label)

    all_data = np.stack(all_data)
    all_labels = np.stack(all_labels)

    data_path = f'{target_path}/data_flocking_{args.randomseed}_{args.dataset_size}_{args.obs_frames}_{args.rollouts}_{N_agent}.npy'
    label_path = f'{target_path}/label_flocking_{args.randomseed}_{args.dataset_size}_{args.obs_frames}_{args.rollouts}_{N_agent}.npy'
    edge_label_path = f'{target_path}/edge_label_flocking_{args.randomseed}_{args.dataset_size}_{args.obs_frames}_{args.rollouts}_{N_agent}.npy'
    
    print(f'all_data shape{all_data.shape}, all_labels shape{all_labels.shape}')
    print(f'{data_path}')
    
    np.save(data_path,all_data)
    np.save(label_path,all_labels)

    
    load_check_data = np.load(data_path)
    load_check_label = np.load(label_path)

    print(f'load_check_data shape{load_check_data.shape}, load_check_label shape{load_check_label.shape}')
