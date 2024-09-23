import numpy as np
from matplotlib.animation import FuncAnimation
from datetime import datetime
from gvae.datasets.small_synth_data import *
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def init_exp_version():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

class show_vicsek_model():
    def __init__(self,pos_read,orient_indiv_read,iteration,trained = False):
        self.pos = pos_read
        self.orient = orient_indiv_read
        self.iteration = iteration
        self.trained = trained
        self.fig, self.ax= plt.subplots(figsize=(6,6))
        self.qv = self.ax.quiver(self.pos[0][:,0], self.pos[0][:,1], np.cos(self.orient[0]), \
                            np.sin(self.orient[0]), self.orient[0], clim=[-np.pi, np.pi])
        self.frame_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes, color='red')

    def update(self,i,ax,pos,orient):
        self.qv.set_offsets(self.pos[i])
        self.qv.set_UVC(np.cos(self.orient[i]), np.sin(self.orient[i]), orient[i])
        self.frame_text.set_text(f'Frame: {i}')
        return self.qv,self.frame_text,

def show_vicsek_model_(pos_read,orient_indiv_read,iteration,trained = False):
    ss = show_vicsek_model(pos_read,orient_indiv_read,iteration,trained)
    anim = FuncAnimation(ss.fig, ss.update, frames=range(iteration),
                         fargs=(ss.ax, pos_read, orient_indiv_read), interval=200, blit=True)
    save_path = init_exp_version()
    anim.save(save_path + ".gif")
    plt.show()

def transform_edges(gt_edges):
    """
    convert N*N adjacent matrix to N*(N-1) graph matrix
    """
    res = torch.zeros(gt_edges.shape[0],gt_edges.shape[1],gt_edges.shape[2]*gt_edges.shape[2] - gt_edges.shape[2])
    num_agent = gt_edges.shape[2]
    k = 0
    for i in range(num_agent):
        for j in range(num_agent):
            if (i is not j):
                res[:,:,k] = gt_edges[:,:,i,j]
                k = k + 1
    return res

def transform_edges_invert(gt_edges,N_agnet):
    """
    convert  N*(N-1) graph matrix to N*N adjacent matrix
    """
    assert N_agnet * (N_agnet-1) == gt_edges.shape[2]
    res = torch.zeros(gt_edges.shape[0],gt_edges.shape[1],N_agnet, N_agnet)
    temp = gt_edges.reshape((gt_edges.shape[0],gt_edges.shape[1],N_agnet,-1))
    for i in range(N_agnet):
        for j in range(N_agnet-1):
            if j >= i:
                res[:,:,i,j+1] = temp[:,:,i,j]
            else:
                res[:,:,i,j] = temp[:,:,i,j]
    return res

def eval_edges(model, data_loader, params,show_res=False):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    eval_metric = params.get('eval_metric')
    num_edge_types = params['num_edge_types']
    skip_first = params['skip_first']
    num_agents = params['num_vars']
    model.eval()
    all_edges = []

    precision = []
    recall =  []
    accuracy = []
    f1 =  []
    cnt = 0

    for batch_ind, batch in enumerate(data_loader):
        inputs = batch[0]
        
        gt_edges_mat = batch[1][:,:,:,-num_agents:].long()
        gt_edges = transform_edges(gt_edges_mat)
       # pdb.set_trace()
        with torch.no_grad():
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                gt_edges = gt_edges.cuda(non_blocking=True)
            _, _, _, edges, _ = model.calculate_loss(inputs, is_train=False, return_logits=True)
            edges = edges.argmax(dim=-1)
            all_edges.append(edges.cpu())
            if len(edges.shape) == 3 and len(gt_edges.shape) == 2:
                gt_edges = gt_edges.unsqueeze(1).expand(gt_edges.size(0), edges.size(1), gt_edges.size(1))
            elif len(gt_edges.shape) == 3 and len(edges.shape) == 2:
                edges = edges.unsqueeze(1).expand(edges.size(0), gt_edges.size(1), edges.size(1))
            if edges.size(1) == gt_edges.size(1) - 1:
                gt_edges = gt_edges[:, :-1]

            edges_pre = transform_edges_invert(edges,num_agents)
           # print(edges_pre.shape,inputs.shape)
            if show_res:
                for batch_id in range(inputs.shape[0]):
                    pos = inputs[batch_id,0:-1,:,0:2].cpu().numpy()
                    vel = inputs[batch_id,0:-1,:,2:4].cpu().numpy()
                    edges_pre_ = edges_pre[batch_id,:,:,:].cpu().numpy()
                    seq_len = np.shape(edges_pre_)[0]
                    check_edge_result(pos,vel,edges_pre_,seq_len,num_agents)

            edges_np = torch.flatten(edges,0).cpu().numpy().astype(np.int8)
            gt_edges_np = torch.flatten(gt_edges,0).cpu().numpy().astype(np.int8)
            accuracy.append(accuracy_score(gt_edges_np,edges_np))
            precision.append(precision_score(gt_edges_np,edges_np,zero_division=0.0))
            recall.append(recall_score(gt_edges_np,edges_np,zero_division=0.0))
            f1.append(f1_score(gt_edges_np,  edges_np ,zero_division=0.0))
            cnt = cnt + 1

    
    accuracy_ = np.mean(accuracy)
    precision_ = np.mean(precision)
    recall_ = np.mean(recall)
    f1_ = np.mean(f1)

    acc_std = np.std(accuracy)
    pre_std = np.std(precision)
    rec_std = np.std(recall)
    f1_std = np.std(f1)
    all_edges = torch.cat(all_edges)
    return accuracy_, precision_, recall_, f1_, acc_std, pre_std, rec_std, f1_std, all_edges


def eval_traj_prediction(model, data_loader, burn_in_steps, forward_pred_steps, params, return_total_errors=False):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    model.eval()
    total_se = 0
    batch_count = 0
    all_errors = []
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch[0]
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
            batch_count += 1
            if return_total_errors:
                all_errors.append(F.mse_loss(model_preds, gt_predictions, reduction='none').view(model_preds.size(0), model_preds.size(1), -1).mean(dim=-1))
            else:
                total_se += F.mse_loss(model_preds, gt_predictions, reduction='none').view(model_preds.size(0), model_preds.size(1), -1).mean(dim=-1).sum(dim=0)
    if return_total_errors:
        return torch.cat(all_errors, dim=0)
    else:
        return total_se / len(data_loader.dataset)
    
def draw_robots(ax,possition_all,velocity_all,N_agent,robot_radius,edge_mat,color = 'b'):
    """
    position_all: N * 2 numpy array, N_agent, Px, Py
    velocity_all: N * 2 numpy array, N_agent, Vx, Vy
    """
    color_vec = ['#DC143C', '#800080', '#7B68EE', '#000080', '#2F4F4F', '#006400', '#8B4513', '#A52A2A', '#000000']
    robot_radius = 0.15
    #print(edge_mat)
    
    for ag in range(N_agent):
        #color_id = ag % len(color_vec)
        #plt.scatter(p_all[ag][0],p_all[ag][1],c=color_vec[color_id],s=1)
        draw_circle = plt.Circle((possition_all[ag][0], possition_all[ag][1]), \
                                robot_radius,fill=False,color=color)
        arr = ax.arrow(possition_all[ag][0], possition_all[ag][1], velocity_all[ag][0], \
                       velocity_all[ag][1],head_width=0.1, head_length=0.1,fc=color, ec=color)
        ax.add_artist(draw_circle)
        text_obj = plt.text(possition_all[ag][0],possition_all[ag][1],str(ag))
        if edge_mat is not None:
            for i in range(N_agent):
                if edge_mat[ag][i] == True and edge_mat[i][ag] == True:
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
        
def check_edge_result(pos_vec,vel_vec,edge_vec,seq_len,N_agent_total,robot_radius = 0.15,perception_radius = 2):
    plt.ion()
    fig,ax = plt.subplots(figsize=(10,10))
    length_out = 3
    x_min = -length_out
    x_max = length_out
    y_min = -length_out
    y_max = length_out
    
    from matplotlib.patches import Rectangle
    
    rectangle_boundary = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='b', facecolor='none')
   # rectangle_unrelevant = Rectangle((x_min_center, y_min_center), x_max_center - x_min_center, y_max_center - y_min_center, linewidth=1, edgecolor='r', facecolor='none')
    ax.set_aspect(1)
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    for t in range(seq_len):
        p_all = pos_vec[t]
        v_all = vel_vec[t]
        edge_c = edge_vec[t].astype(np.bool_)
        #print(edge_c)
        ax.cla()
        ax.add_patch(rectangle_boundary)
        draw_perception_range(ax,p_all,N_agent_total,perception_radius)
        draw_robots(ax,p_all,v_all,N_agent_total,robot_radius,edge_c)
        plt.pause(0.05)
        #plt.waitforbuttonpress()
    plt.ioff()
    plt.show()


### data shape [batch_size, num_timestamp, num_agents, feat_dim]
### edge shape [batch_size, num_timestamp, num_agents,num_agents]

if __name__ == '__main__':
    all_data = np.load('data/swarm/data_flocking_55_50000_24_10.npy')
    all_data = all_data[9]
   # all_data_edge = np.load('data/swarm/label_flocking_55_50000_24_10.npy')
    
    #
    #all_data = np.transpose(all_data,(2,1,0))
    # all_data_p = np.transpose(all_data_p,(2,1,0))
    # all_data = np.concatenate((all_data,all_data_p),axis=1)
   # position_diff = np.diff(all_data, axis=0)
   # print(all_data.shape)
   # print(all_data_p.shape)

   # all_data = np.concatenate((all_data[0:-1,:,:],position_diff),axis=2)
    print(all_data.shape)
   # pdb.set_trace()
    seq_len = 100
    N_agent = 5
    pos = all_data[0:seq_len,-N_agent:,0:2]
    vel = all_data[0:seq_len,-N_agent:,2:]
    edge = np.expand_dims(np.ones(N_agent) - np.eye(N_agent),axis=0).repeat(seq_len,0)

    print(pos.shape)
    print(vel.shape)
    print(edge.shape)
    
   # pdb.set_trace()
    check_edge_result(pos,vel,edge,seq_len,N_agent)


