import numpy as np 
from matplotlib import pyplot as plt 
import vortex_sim as vsim 
# from scipy import interpolate
from matplotlib.animation import FuncAnimation
from numba import njit, prange
import time 



def plotter(name, vortex_final_pos):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.plot(vortex_final_pos[-1][::,0],vortex_final_pos[-1][::,1], ".")
    plt.title(name)
    # plt.savefig(name+".png")
    plt.show()
def linear_point_vortices(n, dt, iter):
    x=np.linspace(-0.5, 0.5, 2*n+1)
    x1=x[::2]
    y=x[1::2]
    gamma= np.array( [ -( np.sqrt(1-4*x1[i+1]**2)- np.sqrt(1-4*x1[i]**2)) for i in range(len(x1)-1) ])
    vortex_position=np.array([[y[i],0.0] for i in range(len(y))])
    vortex_final_pos, tracer_final_pos = vsim.simulation(vortex_position, gamma, dt, iter, 2, np.array([[0,0]]) )

    name="linear_point_vortices_dt"+str(dt)+ "_N"+ str(n)+ "_iter"+ str(iter)
    return(vortex_final_pos, name)
    
def cos1(n,delta, dt, iter):
    x_cos=np.linspace(-np.pi/2, np.pi/2, (2*n)-1)
    x_cos_1=x_cos[::2]
    x_cos_2= x_cos[1::2]
    x_cos_1=np.sin(x_cos_1)*0.5
    y_cos=np.sin(x_cos_2)*0.5
    gamma_cos= np.array( [ -( np.sqrt(1-4*x_cos_1[i+1]**2)- np.sqrt(1-4*x_cos_1[i]**2)) for i in range(len(x_cos_1)-1) ] )

    vortex_position=np.array([[y_cos[i],0.0] for i in range(len(y_cos))])


    vortex_final_pos=vsim.krasny_simulation(vortex_position, gamma_cos, delta, dt, iter)

    name= "Cos1_delta"+ str(delta)+"_N"+str(n) +"_iter"+str(iter)+"_dt"+ str(dt)
    return(vortex_final_pos, name)


def  cos2(n, delta, dt, iter):
    x_cos=np.linspace(0, np.pi, n)
    y2=-np.cos(x_cos)
    gamma2=np.sin(y2)*np.pi/(2*n)
    vortex_position=np.array([[y2[i],0.0] for i in range(len(y2))])
    vortex_final_pos=vsim.krasny_simulation(vortex_position, gamma2, delta, dt, iter)
    name="Cos2_delta"+ str(delta)+"_N"+str(n) +"_iter"+str(iter)+"_dt"+ str(dt)
    return(vortex_final_pos, name)


def Animator_for_fig(name, vortex_final_pos, n, method):
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
    if (method==3):
        ax = plt.axes(xlim=(-1.2, 1.2), ylim=(-0.9, 0.2))
    else:
        ax = plt.axes(xlim=(-0.8, 0.8), ylim=(-0.9, 0.1))
    ax.set_aspect("equal")
    line, = ax.plot([], [], lw=2)
    plt.title(name)
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x= vortex_final_pos[i][::,0]
        y= vortex_final_pos[i][::,1]
        line.set_data(x, y)
        return line,
    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=n-2, interval= 50, blit=True)

    
    anim.save(name+".gif", writer='imagemagick')



def patch(d, J, M):
    patch_pos=np.zeros((2+J*(J+1)*M//2,2))
    counter=0
    for j in prange(1,J+1):
        for i in prange(1, j*M+1):
            x=j*d*np.cos(i*2*np.pi/(j*M))
            y=j*d*np.sin(i*2*np.pi/(j*M))
            patch_pos[counter]=np.array([x,y])
            counter+=1
    print(counter)
    return(patch_pos)

def patch_merge(dist, d, J, M, iter=100, delta=0.05, dt=0.01):
    p= patch(d, J, M)
    p1= p+[dist/2.0, 0]
    p2= p+[-dist/2.0, 0]
    positions= np.concatenate((p2,p1))
    npts=len(positions)
    gamma=np.array([1.0]*npts)
    vortex_final_pos=vsim.krasny_simulation(positions, gamma, delta, dt, iter)
    name= "dist"+str(dist)+"_d"+str(d)+"_J"+str(J)+"_M"+ str(M)+ "_iter"+ str(iter)+"_delta"+str(delta)+"_dt"+str(dt)
    return(vortex_final_pos,name)

    # fig=plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # plt.plot(positions[::,0], positions[::,1], ".")
    # plt.show()
def plot_patch(pos, name):
    npt=len(pos[0])//2
    # plt.plot(pos[0][npt:2*npt:, 0], pos[0][npt:2*npt:,1], "b.")
    # plt.plot(pos[0][:npt:,0], pos[0][:npt:,1], "r.")

    plt.plot(pos[-1][npt:2*npt:, 0], pos[-1][npt:2*npt:,1], "b.")
    plt.plot(pos[-1][:npt:,0], pos[-1][:npt:,1], "r.")
    plt.title(name)
    # plt.plot(vortex_final_pos[50].real, vortex_final_pos[50].imag, ".")
    # plt.show
    plt.show()

def animator_for_patch(pos, dist, name="patch"):
    npts=len(pos[0])//2
    # plt.style.use('seaborn-pastel')
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = plt.axes(xlim=(-dist, dist), ylim=(-dist, dist)) #xlim=(-dist, dist), ylim=(-dist, dist)
    ax.set_aspect("equal")
    # ax.set_facecolor('#000000')
    # ax.patch.set_facecolor('#000000')
    plt.title(name)
    lines=[ax.plot([], [], "r.")[0], ax.plot([], [], "b.")[0]]

    def init():
        for line in lines:
            line.set_data([],[])
        return lines    

    def animate(i):
        lines[0].set_data(pos[i][:npts:,0], pos[i][:npts:,1])
        lines[1].set_data(pos[i][npts:2*npts:,0], pos[i][npts:2*npts:,1])
        return lines


    # def animate(i):
    #     ins=ax.plot(pos[i][:npts:,0], pos[i][:npts:,1], "r.", pos[i][npts:2*npts:,0], pos[i][npts:2*npts:,1],"b." )
    #     return ins
    anim = FuncAnimation(fig, animate, init_func=init,
                                 frames=len(pos)-2, interval= 80, blit=True)

    anim.save(name+".gif", writer='imagemagick')
    # plt.show()



if __name__ == "__main__":
    d=0.3
    dist= 10
    J=10
    M=5
    delta= 0.09
    dt=0.01
    iter=500


    # p=patch(d, 12, 8)
    # pc1=np.array([   i[0]+ 1j*i[1]  for i in p])
    # pt= np.concatenate((pc1-(dist/2 +0j), pc1+(dist/2 +0j)))
    # gamma=np.array([1.0]*npts)

    n=100
    x_cos=np.linspace(0, np.pi, n)
    y2=-np.cos(x_cos)
    pt=np.array([i+0j for i in y2])
    gamma=np.sin(y2)*np.pi/(2*n)


    # npts=len(pt)
    
    vortex_final_pos = vsim.krasny_sim_comp(pt, delta, gamma, dt, iter)
    # plt.plot(vortex_final_pos[0].real, vortex_final_pos[0].imag, ".")
    plt.plot(vortex_final_pos[iter].real, vortex_final_pos[iter].imag)
    plt.show()

    # vortex_final_pos, name = patch_merge(dist, d, J, M, iter, delta, dt)
    # plot_patch(vortex_final_pos, name)
























# x=np.linspace(-0.5, 0.5, n)
# gamma =np.array([ -( np.sqrt(1-4*x[i+1]**2)- np.sqrt(1-4*x[i]**2)) for i in range(len(x)-1) ])

# y=np.linspace(dx-0.5, 0.5-dx, n-1)

# x_cos=np.linspace(-np.pi/2, np.pi/2, (2*n)-1)
# x_cos_1=x_cos[::2]
# x_cos_2= x_cos[1::2]
# x_cos_1=np.sin(x_cos_1)*0.5

# y_cos=np.sin(x_cos_2)*0.5

# gamma_cos= np.array( [ -( np.sqrt(1-4*x_cos_1[i+1]**2)- np.sqrt(1-4*x_cos_1[i]**2)) for i in range(len(x_cos_1)-1) ] )

# y2=-np.cos(x_cos[::2])
# gamma2=np.sin(y2)*np.pi/(2*n)

# vortex_position=np.array([[y_cos[i],0.0] for i in range(len(y_cos))])

# vortex_final_pos=vsim.krasny_simulation(vortex_position, gamma_cos, delta, dt, iter)

# fig=plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# # plt.plot(vortex_final_pos[::,::,0], vortex_final_pos[::,::,1])

# totalx=np.hstack((vortex_final_pos[-1][::,0], -vortex_final_pos[-1][::,0]) )
# totaly= np.hstack((vortex_final_pos[-1][::,1], vortex_final_pos[-1][::,1]))

# plt.plot(totalx,totaly)
# plt.title("Cos_delta"+ str(delta)+"_N"+str(n) +"_iter"+str(iter)+"_dt"+ str(dt))
# plt.savefig("Cos_delta"+ str(delta)+"_N"+str(n) +"_iter"+str(iter)+"_dt"+ str(dt)+".png")
# plt.show()






# plt.style.use('seaborn-pastel')
# fig = plt.figure()
# ax = plt.axes(xlim=(-0.8, 0.8), ylim=(-0.9, 0.2))
# ax.set_aspect("equal")
# line, = ax.plot([], [], lw=2)

# def init():
#     line.set_data([], [])
#     return line,

# def animate(i):
#     x= vortex_final_pos[i][::,0]
#     y= vortex_final_pos[i][::,1]
#     line.set_data(x, y)
#     return line,
# anim = FuncAnimation(fig, animate, init_func=init,
#                                frames=n-1, interval=n-1, blit=True)

# anim.save("Cos_delta"+ str(delta)+"_N"+str(n) +"_iter"+str(iter)+"_dt"+ str(dt)+".gif", writer='imagemagick')








# vortex_final_pos2, tracer_final_pos = vsim.simulation(vortex_position, gamma, 0.8/100, 100, 2, np.array([[0,0]]) )

# fig=plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# # plt.plot(vortex_final_pos2[::,::,0], vortex_final_pos[::,::,1])

# plt.plot(vortex_final_pos2[-1][::,0],vortex_fina2l_pos[-1][::,1])
# plt.savefig("dirac.png")

# plt.plot(y2, gamma2, ".");
# plt.plot(y2, [0]*len(y2), ".")


