import numpy as np
import matplotlib.pyplot as plt
import pdb
from common_utils import *
import torch

def leader_follower_control_test():
    leader_start  = np.array([3,3],dtype=float)
    leader_end = np.array([10,3],dtype=float)
    follower_start = np.array([2,2],dtype=float)

    v_leader_0 = np.array([-0.2,-0.2],dtype=float)
    v_follower_0 = np.array([0,0],dtype=float)

    v_leader = v_leader_0
    v_f = v_follower_0
    alpha_leader = 0.1

    pos_leader = leader_start
    pos_follower = follower_start
    plt.subplots()
    plt.xlim(-1,20)
    plt.ylim(-1,20)
    plt.ion()
    alpha_f = 0.001
    beta_f = 1
    for t in range(200):
        ## updaet position
        plt.scatter(pos_leader[0],pos_leader[1],c='b')
        plt.scatter(pos_follower[0],pos_follower[1],c='r')
        
        #############
        pos_leader_next = pos_leader + v_leader
        pos_follower_next = pos_follower + v_f
        v_f_next = v_f + alpha_f * (pos_leader + [-1,-1] - pos_follower) + beta_f * (v_leader - v_f)
        v_l_next = alpha_leader * (leader_end - pos_leader)
        pos_leader = pos_leader_next
        v_leader = v_l_next
        pos_follower = pos_follower_next
        v_f = v_f_next
        #############
        plt.pause(0.1)
    plt.ioff()
    plt.show()


def sin_trajec_test():
    N_agent_main = 6
    start_pt = torch.tensor([[0.1,0.1],
                            [0.1,0.2],
                            [0.2,0.1],
                            [0.2,0.2],
                            [0.3,0.1],
                            [0.3,0.2]]) * 3

    start_v = torch.rand(N_agent_main,2) * 3
    disturb_id = []
    
    v_all = start_v
    p_all = start_pt
    
    #################  check leader trajectory ################# 
    # plt.subplots()
    # pt = start_pt[0]
    # for t in range(100):
    #     pt = pt + torch.tensor(compute_leader_v(pt[0]))
    #     plt.scatter(pt[0],pt[1])
    # plt.show()
    ############################################################

    mig_target = start_pt + torch.tensor((10,0))
    edge_mat = np.random.choice([0,1],
                                size=(N_agent_main, N_agent_main))

    
    edge_mat = np.tril(edge_mat) + np.tril(edge_mat, -1).T
    np.fill_diagonal(edge_mat, 0)
    edge_mat = torch.tensor(edge_mat).bool()
    print(edge_mat)
    #pdb.set_trace()
    
    edge_mat_full = ~torch.eye(6).bool()
    edge_mat_input = edge_mat
    c = (-1) * (start_pt.unsqueeze(1) - start_pt.unsqueeze(0))
    plt.subplots()
    plt.xlim(-2,20)
    plt.ylim(-2,2)
    plt.ion()
    color_vec = ['#DC143C', '#800080', '#7B68EE', '#000080', '#2F4F4F', '#006400', '#8B4513', '#A52A2A', '#000000']
    text_objs = []

    pos_vec = []
    vel_vec = []
    
    seq_len = 100
    for t in range(seq_len):
        v_all = torch.zeros_like(v_all)
        c = (-1) * (p_all.unsqueeze(1) - p_all.unsqueeze(0))
        c = c + torch.rand(c.shape) * 0 ## Add noise
        mig_target = p_all[0]
        mig_tar_current = mig_target - p_all.unsqueeze(0)
        for i in range(N_agent_main):
            if i in disturb_id:
                continue
            if i == 0: ### leader
                v = torch.tensor(compute_sin_leader_v(p_all[i][0]))
            else:
                v = compute_indivisual_v(i,c,mig_tar_current.squeeze(0)[i],edge_mat_input)
            v_all[i] = v_all[i] + v
        p_all = p_all + v_all
        #print(p_all.shape,v_all.shape)
        pos_vec.append(p_all.numpy())
        vel_vec.append(v_all.numpy())
        
        for text_obj in text_objs:
            text_obj.remove()
        text_objs = []
        for ag in range(N_agent_main):
            color_id = ag % len(color_vec)
            plt.scatter(p_all[ag][0],p_all[ag][1],c=color_vec[color_id],s=1)
            text_obj = plt.text(p_all[ag][0],p_all[ag][1],str(ag))
            text_objs.append(text_obj)
        plt.pause(0.1)
        
        
    loc_train = np.array(pos_vec)
    vel_train = np.array(vel_vec)
    edge_train = edge_mat.numpy()
    
    print("train loc data shape:",np.shape(loc_train))
    print("train vel data shape:",np.shape(vel_train))
    print("edge data shape:",np.shape(edge_train))
    

    ## save file: loc_MODE_AGENT-NUMBER_OBSERVE-FRAME_DATA-NUM.pkl
    loc_path = f'dataset/formation/loc_{N_agent_main}_{seq_len}.pkl'
    vel_path = f'dataset/formation/vel_{N_agent_main}_{seq_len}.pkl'
    edge_path = f'dataset/formation/edge_{N_agent_main}_{seq_len}.pkl'
    
    #'data/loc_' + post_fix + '_'  + str() + '.pkl'
    with open(loc_path, 'wb') as f:
        pickle.dump(loc_train, f)
    with open(vel_path, 'wb') as f:
        pickle.dump(vel_train, f)
    with open(edge_path, 'wb') as f:
        pickle.dump(edge_train, f)
    suffix = 'formation'
    np.save('loc_train' + suffix + '.npy', loc_train)
     #   pdb.set_trace()
        
        #############################
        # for text_obj in text_objs:
        #     text_obj.remove()
        # text_objs = []
        # for ag in range(N_agent_main):
        #     color_id = ag % len(color_vec)
        #     plt.scatter(p_all[ag][0],p_all[ag][1],c=color_vec[color_id],s=1)
        #     text_obj = plt.text(p_all[ag][0],p_all[ag][1],str(ag))
        #     text_objs.append(text_obj)
        # plt.pause(0.1)
        ################################
    plt.ioff()
    plt.show()


def combine_edge(edge_group0,edge_group1):
    N0 = np.shape(edge_group0)[0]
    N1 = np.shape(edge_group1)[1]
    N = N0 + N1
    res = torch.zeros((N,N))
    res[0:N0,0:N0] = edge_group0
    res[N0:N,N0:N] = edge_group1
    return res

def generate_random_graph(N_agent,mode = 0):
    """
    mode 0: fully connected
    mode 1: leader randomly connected, others fully connected
    mode 2: all randomly connected
    """
    edge_mat = None
    
    if mode == 0:
        edge_mat = ~torch.eye(N_agent).bool()
    elif mode == 1:
        edge_leader_false = np.random.choice([0,1],
                                size=(N_agent,))
    
        edge_mat = ~torch.eye(N_agent).bool()
        edge_mat[0,:] = torch.tensor(edge_leader_false)
        edge_mat[:,0] = torch.tensor(edge_leader_false)
        edge_mat[0,0] = False
        if (edge_mat[0,:] == True).any() == False:
            edge_mat[0,1] = True
            edge_mat[1,0] = True
    elif mode == 2:
        edge_mat = np.random.choice([0,1],
                                size=(N_agent, N_agent))
        edge_mat = np.tril(edge_mat) + np.tril(edge_mat, -1).T
        np.fill_diagonal(edge_mat, 0)
        edge_mat = torch.tensor(edge_mat).bool()
        #edge_mat[0,1:] = True
        #edge_mat[1:,0] = True
    
    return edge_mat

def compute_circle_leader_v(t):
    r = 2.5
    omega = 0.02
    vx = - omega * r * np.sin(omega * t)
    vy =  omega * r * np.cos(omega * t)
    v = np.array([vx,vy])
    if np.linalg.norm(v) > 0.4:
        v = v / np.linalg.norm(v) * 0.4
    return v

def compute_retangular_leader_v(t,seq_len,v_norm):
    v = np.array([0,0])
    if t < seq_len / 8:
        v = np.array([0,1])
    elif t >= seq_len / 8 and t < 3 * seq_len / 8:
        v = np.array([-1,0])
    elif t >= 3 * seq_len / 8 and t < 5 * seq_len / 8:
        v = np.array([0,-1])
    elif t >= 5 * seq_len / 8 and t < 7 * seq_len / 8:
        v = np.array([1,0])
    elif t >= 7 * seq_len / 8 and t < seq_len:
        v = np.array([0,1])
    elif t == seq_len:
        v = np.array([0,0])
    else:
        print(f"wrong t input:{t},seq_len {seq_len} ")
    v = v * v_norm
    return v

def compute_sin_leader_v(x):
    vx = 0.1
    vy = 0.075 * np.cos(np.pi / 2 * x)
    return (vx,vy)

def compute_indivisual_v(index,relative_pos,r_mig,edge_mat):
    """
    Use Reynold flocking dynamics
    """
    N_neighbor = len(relative_pos)
    v_sep = 0
    k_sep = 1

    for i in range(N_neighbor):
        if i == index:
            continue
        v_sep = v_sep + relative_pos[index][i] / np.linalg.norm(relative_pos[index][i]) * (-1)
    v_sep = v_sep / (N_neighbor-1) * k_sep

    v_coh = 0
    k_coh = 0.6
    
    for i in range(N_neighbor):
        if edge_mat[index][i] == False:
            continue
        v_coh = v_coh + relative_pos[index][i]
    v_coh = v_coh / (N_neighbor-1) * k_coh

    v_mig = 0
    k_mig = 0.05
    v_mig = k_mig * r_mig / np.linalg.norm(r_mig)
    v = v_sep + v_coh + v_mig
    if torch.norm(v) >= 0.02:
        v = v / torch.norm(v) * 0.02
    return v

def compute_v_PD(target_x,target_y,px,py,vx,vy,vx_last,vy_last):
    target_x = target_x.double()
    target_y = target_y.double()
    px = px.double()
    py = py.double()
    vx = vx.double()
    vy = vy.double()
    vx_last = vx.double()
    vy_last = vy.double()
    
    kp = 0.04
    kv = 0.1
    vx_next = kp * (target_x - px) + kv* (vx - vx_last)
    vy_next = kp * (target_y - py) + kv* (vy - vy_last)
    
    theta_next = np.arctan2(vy_next,vx_next)
    theta_current = np.arctan2(vy,vx)

    v = np.array([vx_next,vy_next])
    if np.linalg.norm(v) > 0.05:
        v = v / np.linalg.norm(v) * 0.05
    return v

def compute_indivisual_v_v2(index,relative_pos,r_mig,edge_mat,v_last,k_sep = 0.59,k_mig = 0.01,k = 0.06):
    """
    Use unified flocking dynamics
    """
    N_neighbor = len(relative_pos)
    v = 0
    k = 0.06
    k_last = 0.95
    A = (k_sep ** 3)
    for i in range(N_neighbor):
        if edge_mat[index][i] == False:
            continue
        norm_r = np.linalg.norm(relative_pos[index][i])
        left = 1 - A / (norm_r ** 3)
        v = v + left * relative_pos[index][i]
    v = v * k
    v_mig = k_mig * r_mig / np.linalg.norm(r_mig)
    if edge_mat[index][0] == False:
        v_mig = v_mig * 0
    v = v + v_mig + k_last * v_last
    if np.linalg.norm(v) > 0.1:
        v = v / np.linalg.norm(v) * 0.1
    return v

def compute_individual_v_unrelevant(index,
                                    relative_pos,
                                    relative_boundary,
                                    current_v,
                                    r_mig,
                                    edge_mat):
    N_neighbor = len(relative_pos)
    N_boundary = relative_boundary.shape[1]
    v = torch.zeros((2,))
    
    v_sep = torch.zeros((2,))
    k_sep = 0.1
    for i in range(N_neighbor):
        if edge_mat[index][i] == False:
            continue
        v_sep = v_sep + relative_pos[index][i] / np.linalg.norm(relative_pos[index][i]) * (-1)
    v_sep = v_sep / (N_neighbor) * k_sep
    
    v_sep_boundary = 0
    k_sep_boundary = 0.01
    for i in range(N_boundary):
        v_sep_boundary = v_sep_boundary + current_v * relative_boundary[index][i] * (-1)
    v_sep_boundary = v_sep_boundary * k_sep_boundary
    v_rand = current_v + (torch.rand((2,)) - 0.5) * 0.1
    
    v_mig = 0
    k_mig = 0.1
    v_mig = k_mig * r_mig / np.linalg.norm(r_mig)

    v = v_sep + v_sep_boundary + v_mig
   ## print("v_sep:",v_sep)
   # print("N_boundary:",N_boundary)
   # print("relative_boundary:",relative_boundary)
   # print("v_sep_boundary:",v_sep_boundary)

    ######## bounce back ###############

    if relative_boundary[index][0] < 0.15:
        v[0] = current_v[0]
        v[1] = -current_v[1] * 2
    if relative_boundary[index][1] < 0.15:
        v[0] = -current_v[0]* 2
        v[1] = current_v[1]
    if relative_boundary[index][2] < 0.15:
        v[0] = current_v[0]
        v[1] = -current_v[1]* 2
    if relative_boundary[index][3] < 0.15:
        v[0] = -current_v[0] * 2
        v[1] = current_v[1]

    if torch.norm(v) >= 0.2:
        v = v / torch.norm(v) * 0.2
    return v

def compute_dist_boundary(p_all_unrelevant,x_max,x_min,y_max,y_min):
    N_unrelevant = p_all_unrelevant.shape[0]
    dist_boundary = torch.zeros((N_unrelevant,4))
    for i in range(N_unrelevant):
        dist_boundary[i][0] = torch.abs(y_min - p_all_unrelevant[i][1])
        dist_boundary[i][1] = torch.abs(x_max - p_all_unrelevant[i][0])
        dist_boundary[i][2] = torch.abs(y_max - p_all_unrelevant[i][1])
        dist_boundary[i][3] = torch.abs(x_min - p_all_unrelevant[i][0])
    return dist_boundary

def generate_leader_rec():
    """
    generate rectangular trajectory
    """
    width = 10
    height = 5
    boundary_dist = 2
    amp = 0.3
    T = 2
    start_pt = np.array([1,1])
    start_v = np.array([0,0])
    v = start_v
    pos = start_pt
    plt.subplots()
    plt.xlim(-1,20)
    plt.ylim(-1,20)
    plt.ion()
    for t in range(100):
        mode = -1
        if t < 25:
            vx = 0.4
            vy = amp*np.cos(2 * np.pi / T * pos[0])
            v = np.array([vx,vy])
            mode = 0
        elif t >= 25 and t < 50:
            vx = amp*np.cos(2 * np.pi / T * pos[1])
            vy = 0.4
            v = np.array([vx,vy])
            mode = 1
        elif t >= 50 and t < 75:
            vx = -0.4 
            vy = amp*np.cos(2 * np.pi / T * pos[0])
            v = np.array([vx,vy])
            mode = 2
        elif t >= 75 and t < 100:
            vx = amp*np.cos(2 * np.pi / T * pos[1])
            vy = -0.4
            v = np.array([vx,vy])
            mode = 2
        pos = pos + v
        print(v,pos,mode)
        plt.scatter(pos[0],pos[1],c='b')
        plt.pause(0.1)
    plt.ioff()
    plt.show()

def generate_leader_circular():
    """
    generate circular trajectory
    """
    width = 10
    height = 5
    boundary_dist = 2
    center = (5,5)
    omega = 0.1
    T = 2
    start_pt = np.array([5,5])
    start_v = np.array([0,0])
    v = start_v
    pos = start_pt
    plt.subplots()
    plt.xlim(-1,10)
    plt.ylim(-1,10)
    plt.ion()
    for t in range(100):
        theta = omega * t
        vx = -omega * np.sin(omega * t)
        vy = omega * np.cos(omega * t)
        v = np.array([vx,vy])
        pos = pos + v
        plt.scatter(pos[0],pos[1],c='b')
        plt.pause(0.1)
    plt.ioff()
    plt.show()

def draw_robots(ax,possition_all,velocity_all,N_agent,robot_radius,edge_mat,color = 'b'):
    """
    position_all: N * 2 numpy array, N_agent, Px, Py
    velocity_all: N * 2 numpy array, N_agent, Vx, Vy
    """
    color_vec = ['#DC143C', '#800080', '#7B68EE', '#000080', '#2F4F4F', '#006400', '#8B4513', '#A52A2A', '#000000']
    robot_radius = 0.15
    for ag in range(N_agent):
        #color_id = ag % len(color_vec)
        #plt.scatter(p_all[ag][0],p_all[ag][1],c=color_vec[color_id],s=1)
        draw_circle = plt.Circle((possition_all[ag][0], possition_all[ag][1]), \
                                robot_radius,fill=False,color=color)
        arr = ax.arrow(possition_all[ag][0], possition_all[ag][1], velocity_all[ag][0], \
                       velocity_all[ag][1],head_width=0.1, head_length=0.1,fc=color, ec=color)
        ax.add_artist(draw_circle)
        text_obj = plt.text(possition_all[ag][0],possition_all[ag][1],str(ag))
        for i in range(N_agent):
            if i != ag and edge_mat[ag][i] == True:
                plt.plot([possition_all[ag][0], possition_all[i][0]], [possition_all[ag][1], possition_all[i][1]], color='r')

def draw_perception_range(ax,possition_all,N_agent,percep_r,color = 'b'):
    """
    position_all: N * 2 numpy array, N_agent, Px, Py
    velocity_all: N * 2 numpy array, N_agent, Vx, Vy
    """
    color_vec = ['#DC143C', '#800080', '#7B68EE', '#000080', '#2F4F4F', '#006400', '#8B4513', '#A52A2A', '#000000']
    for ag in range(N_agent):
        if ag == 0:
            color = 'red'
        else:
            color = 'green'
        draw_circle = plt.Circle((possition_all[ag][0], possition_all[ag][1]), \
                                percep_r,fill=False,color=color)
        ax.add_artist(draw_circle)

def generate_initial_pos(N_agent,robot_radius,x_max,x_min,y_max,y_min):
    pos = torch.rand((N_agent,2))
    pos[:,0] = pos[:,0] * (x_max - x_min)
    pos[:,0] = pos[:,0] + x_min
    pos[:,1] = pos[:,1] * (y_max - y_min)
    pos[:,1] = pos[:,1] + y_min
    while True:
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        dd = torch.norm(dist,dim=2) + torch.eye(N_agent)
        if (dd < robot_radius * 2 + 0.1).any():
            pos = torch.rand((N_agent,2))
            pos[:,0] = pos[:,0] * (x_max - x_min)
            pos[:,0] = pos[:,0] + x_min
            pos[:,1] = pos[:,1] * (y_max - y_min)
            pos[:,1] = pos[:,1] + y_min
        else:
            break
    return pos

import math
def generate_main_initial_pos(N_agent,robot_radius,x_max,x_min,y_max,y_min,R = 3):
    theta = (torch.rand((1,)) - 0.5) * math.pi
    R = R + (torch.rand((1,)) - 0.5) * 0.5
    area_x = R * torch.cos(theta)
    area_y = R * torch.sin(theta)
    pos = torch.rand((N_agent,2))
    pos[:,0] = pos[:,0] * (x_max - x_min)
    pos[:,0] = pos[:,0] + x_min + area_x
    pos[:,1] = pos[:,1] * (y_max - y_min)
    pos[:,1] = pos[:,1] + y_min + area_y
    while True:
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        dd = torch.norm(dist,dim=2) + torch.eye(N_agent)
        if (dd < robot_radius * 2 + 0.3).any():
            pos = torch.rand((N_agent,2))
            pos[:,0] = pos[:,0] * (x_max - x_min)
            pos[:,0] = pos[:,0] + x_min + area_x
            pos[:,1] = pos[:,1] * (y_max - y_min)
            pos[:,1] = pos[:,1] + y_min + area_y
        else:
            break
    pos[0,0] = area_x
    pos[0,1] = area_y
    return pos


if __name__ == '__main__':
    N_agent = 4
    robot_radius = 0.3
    x_max = 1
    x_min = -1
    y_max = 1
    y_min = -1
    pos = generate_main_initial_pos(N_agent,robot_radius,x_max,x_min,y_max,y_min)
    print(pos)