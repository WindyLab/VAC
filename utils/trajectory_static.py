import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
from common_utils import *
import torch
from formation_utils import *
from matplotlib.patches import Rectangle
import argparse
from tqdm import tqdm
import os
import shutil


def trajec_test(args):
    ############  Parameters  ##################
    N_agent_main = args.num_main
    N_agent_unreleve = args.num_unre
    interactive_show = args.intershow
    save_ani = args.save_ani
    perception_radius = args.perception_radius
    mode = args.motion_mode
    robot_radius = 0.15
    seq_len = args.obs_frames + args.rollouts
    obs_frames = args.obs_frames
    rollouts = args.rollouts
    start_pt = torch.tensor([  [2.8,     0.6 - 0.6],
                               [2.778,  -0.4 - 0.5],
                               [2.35,    0.4 - 0.6]]  )

    start_v = torch.rand(N_agent_main,2) * 0.2
    start_v_unrele = torch.rand(N_agent_unreleve,2) * 0.2

    start_theta = torch.zeros(N_agent_main,)

    start_pt_unrelevant = torch.tensor([ [0,       0 - 0.1],
                                         [-0.4,    0 - 0.1] ] )

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

    theta_start_leader_pt = torch.rand(N_agent_main,2)
    
    start_pt = generate_main_initial_pos(N_agent_main,robot_radius,-1,1,-1,1,R=2)
    #start_pt[0] = torch.tensor([2.8,0.2])
    start_pt_unrelevant = generate_initial_pos(     N_agent_unreleve,
                                                    robot_radius,
                                                    x_max_center,
                                                    x_min_center,
                                                    y_max_center,
                                                    y_min_center)
    ################################################################
    v_all = start_v
    p_all = start_pt
    theta_all = start_theta
    v_all_unrelevant = torch.rand(N_agent_unreleve,2) * 0.75
    p_all_unrelevant = start_pt_unrelevant
    theta_all_unrelevant = torch.zeros(N_agent_unreleve,)
    
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

    mig_target = start_pt + torch.tensor((10,0))
    edge_mat_input  = generate_random_graph(N_agent_main,      mode=2)
    edge_mat_unrele = generate_random_graph(N_agent_unreleve,  mode=2)
    #print("edge_mat_input:\n",edge_mat_input)
    #print("edge_mat_input:\n",edge_mat_unrele)

    pos_vec = []
    vel_vec = []
    theta_vec = []
    theta_unrelevant_vec = []
    pos_unrelevant_vec = []
    vel_unrelevant_vec = []
    random_walk_target = (torch.rand((2,)) - 0.5)  * (length_center * random_target_ratio)

    for t in range(seq_len):
        ##########  Main part  ##############
        if t > 0:
            v_all = torch.zeros_like(v_all)
            theta_all = torch.zeros_like(theta_all)
        relative_pos = (-1) * (p_all.unsqueeze(1) - p_all.unsqueeze(0))
        relative_pos = relative_pos + torch.rand(relative_pos.shape) * 0.01 ## Add noise
        relative_dist = torch.norm(relative_pos,dim=2) + torch.eye(N_agent_main)
        if args.collision_warning and (relative_dist < 2 * robot_radius ).any():
           print("warning!!collision outside!! frame:",t)
           print(relative_dist)
           print('\n')
        mig_target = p_all[0]
        mig_tar_current = mig_target - p_all.unsqueeze(0)
        for i in range(N_agent_main):
            if i in disturb_id:
                continue
            if i == 0: ### leader
                #v = torch.tensor(compute_circle_leader_v(t))
                if mode == 0:
                    v_norm = 4 / seq_len
                    v = torch.tensor(compute_retangular_leader_v(t,seq_len,v_norm))
                elif mode == 1:
                    v = torch.tensor(compute_circle_leader_v(t))
            else:
               v = compute_indivisual_v_v2(i,relative_pos,mig_tar_current.squeeze(0)[i],edge_mat_input,
                                           k_sep = 0.5,k_mig= 0.03)  ##
               # v = compute_indivisual_v(i,relative_pos,mig_tar_current.squeeze(0)[i],edge_mat_input)
            v_all[i] = v
            raw_az = torch.atan2(v[1],v[0]).item()
            theta_all[i] = raw_az
            
        p_all = p_all + v_all
        pos_vec.append(p_all.numpy())
        vel_vec.append(v_all.numpy())
        theta_vec.append(theta_all.numpy())

        ####### Unrelevant Part ################
        v_all_unrelevant_new = torch.zeros_like(v_all_unrelevant)
        relative_pos_unrelevant = (-1) * (p_all_unrelevant.unsqueeze(1) - p_all_unrelevant.unsqueeze(0))
        relative_pos_unrelevant = relative_pos_unrelevant + torch.rand(p_all_unrelevant.shape) * 0.001 ## Add noise
        theta_all_unrelevant = torch.zeros_like(theta_all_unrelevant)
        for i in range(N_agent_unreleve):
            if t > 0:
                v_last_x = vel_unrelevant_vec[t-1][i][0]
                v_last_y = vel_unrelevant_vec[t-1][i][1]
            else:
                v_last_x = start_v_unrele[i][0]
                v_last_y = start_v_unrele[i][1]
            if i == 0:
                temp = compute_v_PD (                   random_walk_target[0],
                                                        random_walk_target[1],
                                                        p_all_unrelevant[i][0],
                                                        p_all_unrelevant[i][1],
                                                        v_all_unrelevant[i][0],
                                                        v_all_unrelevant[i][1],
                                                        v_last_x,
                                                        v_last_y )
                v_all_unrelevant_new[i] = torch.tensor(temp)
            else:
                v_all_unrelevant_new[i] = compute_indivisual_v_v2(i,
                                                                  relative_pos_unrelevant,
                                                                  p_all_unrelevant[0],
                                                                  edge_mat_unrele,
                                                                  k_sep= 0.5,
                                                                  k_mig=0.005,
                                                                  k = 0.006)
            raw_az = torch.atan2(v_all_unrelevant_new[i][1],v_all_unrelevant_new[i][0]).item() - torch.pi / 2
            theta_all_unrelevant[i] = raw_az

        if torch.norm(p_all_unrelevant[0] - random_walk_target) < 0.03:
            #print("target reached!!!")
            random_walk_target = (torch.rand((2,)) - 0.5) * (length_center * random_target_ratio)
            #print("new target:",random_walk_target)
        
        relative_dist_unrelevant = torch.norm(relative_pos_unrelevant,dim=2) + torch.eye(N_agent_unreleve)
        if args.collision_warning and (relative_dist_unrelevant < 2 * robot_radius + 0.02).any():
            print("warning!!collision in center!!frame:",t)
            print(relative_dist_unrelevant)
            print('\n')
        p_all_unrelevant = p_all_unrelevant + v_all_unrelevant_new
        v_all_unrelevant = v_all_unrelevant_new
        pos_unrelevant_vec.append(p_all_unrelevant.numpy())
        vel_unrelevant_vec.append(v_all_unrelevant.numpy())
        theta_unrelevant_vec.append(theta_all_unrelevant.numpy())

    theta_vec = np.expand_dims(theta_vec,axis=2)
    theta_unrelevant_vec = np.expand_dims(theta_unrelevant_vec,axis=2)
    p_all_save = np.concatenate((pos_vec,vel_vec),                              axis=2)
    p_all_unrele_save = np.concatenate((pos_unrelevant_vec,vel_unrelevant_vec),         axis=2)

    final_pos = np.concatenate((p_all_save,p_all_unrele_save),axis=1)
    current_data  =  final_pos[0:obs_frames,:,0:args.feat_dim]
    current_label =  final_pos[obs_frames:,:,0:args.feat_dim]
    edge = combine_edge(edge_mat_input,edge_mat_unrele)

    current_label = torch.tensor(current_label)
    edge = edge.unsqueeze(0).repeat(rollouts,1,1)

    current_label = torch.cat((current_label,edge),dim=2).numpy()
  #  pdb.set_trace()


    ############check angle#############
    # fig1,ax1 = plt.subplots(figsize=(10,10))
    # ax1.plot(np.array(theta_vec)[:,0] * 180 / np.pi)
    # plt.show()
    ##############

    if save_ani:
        import matplotlib.animation as animation
        def update(t):
            p_all = pos_vec[t]
            v_all = vel_vec[t]
            p_all_unrelevant = pos_unrelevant_vec[t]
            v_all_unrelevant = vel_unrelevant_vec[t]
            ax.cla()
            ax.add_patch(rectangle_boundary)
            ax.add_patch(rectangle_unrelevant)
            draw_robots(ax,p_all,v_all,N_agent_main,robot_radius,edge_mat_input)
            draw_perception_range(ax,p_all,N_agent_main,perception_radius)
            draw_robots(ax,p_all_unrelevant,v_all_unrelevant,N_agent_unreleve,robot_radius,edge_mat_unrele,'red')
        ani = animation.FuncAnimation(fig, update,frames=1000)
        ani.save('Liss.gif',fps = 100)

    if interactive_show:
        plt.ion()
        fig,ax = plt.subplots(figsize=(10,10))
        rectangle_boundary = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='b', facecolor='none')
        rectangle_unrelevant = Rectangle((x_min_center, y_min_center), x_max_center - x_min_center, y_max_center - y_min_center, linewidth=1, edgecolor='r', facecolor='none')
        ax.set_aspect(1)
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        for t in range(seq_len):
            p_all = pos_vec[t]
            v_all = vel_vec[t]
            p_all_unrelevant = pos_unrelevant_vec[t]
            v_all_unrelevant = vel_unrelevant_vec[t]
            ax.cla()
            ax.add_patch(rectangle_boundary)
            ax.add_patch(rectangle_unrelevant)
            draw_robots(ax,p_all,v_all,N_agent_main,robot_radius,edge_mat_input)
            draw_perception_range(ax,p_all,N_agent_main,perception_radius)
            draw_robots(ax,p_all_unrelevant,v_all_unrelevant,N_agent_unreleve,robot_radius,edge_mat_unrele,'red')
            if interactive_show:
                plt.pause(0.05)
        if interactive_show:
            plt.ioff()
        plt.show()
    return current_data, current_label