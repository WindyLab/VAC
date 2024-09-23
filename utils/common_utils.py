import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pdb
import pickle
import copy
from datetime import datetime
import scipy.io

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
        self.frame_text.set_text(f'Frame: {i}')  # 更新帧数文本
        return self.qv,self.frame_text,

def show_vicsek_model_(pos_read,orient_indiv_read,iteration,trained = False):
    ss = show_vicsek_model(pos_read,orient_indiv_read,iteration,trained)
    anim = FuncAnimation(ss.fig, ss.update, frames=range(iteration),
                         fargs=(ss.ax, pos_read, orient_indiv_read), interval=200, blit=True)
    save_path = init_exp_version()
    anim.save(save_path + ".gif")
    plt.show()
    
def read_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat
