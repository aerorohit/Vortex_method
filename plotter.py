import numpy as np 
import matplotlib.pyplot as plt 
import time
import sys
from numba import njit, prange
import vortex_sim as vsim
import cmath
from matplotlib.animation import FuncAnimation
# plt.rcParams["figure.figsize"]=15, 15


def animator_for_patch(pos1, pos2, dist, name):
    # plt.style.use('seaborn-pastel')
    plt.style.use("dark_background")
    
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(xlim=(-2, dist), ylim=(-6, 6)) #xlim=(-dist, dist), ylim=(-dist, dist)
    ax.set_aspect("equal")
    plt.title(name)
    tcir=np.linspace(0,2*np.pi, 100)
    plt.plot(np.cos(tcir), np.sin(tcir))
    # ax.set_facecolor('#000000')
    # ax.patch.set_facecolor('#000000')
    lines=[ax.plot([], [],"r.", ms=2)[0], ax.plot([], [], "b.", ms=2)[0]]

    def init():
        for line in lines:
            line.set_data([],[])
        return lines    

    def animate(i):
        lines[0].set_data(pos1[i].real, pos1[i].imag)
        lines[1].set_data(pos2[i].real, pos2[i].imag)
        return lines


    
    anim = FuncAnimation(fig, animate, init_func=init,
                                 frames=len(pos1)-2, interval= 80, blit=True)

    anim.save("rvm_gif/"+name+".gif", writer='ffmpeg') #replace name
    plt.show()


a=np.load("rvm_gif/all_pos_n.npy", allow_pickle= True)
b=np.load("rvm_gif/all_pos_p.npy", allow_pickle= True)

animator_for_patch(a, b, 25, "master")