import numpy as np 
import matplotlib.pyplot as plt 
import time
import sys
from numba import njit, prange


def time_period(vort_pos):
    r_vec= vort_pos[0]-vort_pos[1]
    r=np.linalg.norm(r_vec)
    T=2*np.pi*np.pi*r*r
    if len(vort_pos)==2:
        return(T)
    if len(vort_pos)==3:
        return(T*2.0/3.0)
    if len(vort_pos)==4:
        return(T/2.0)
     



def true_soln(vort_pos_2, time):
    len_vort=len(vort_pos_2)
    if len_vort>4:
        sys.exit("Error: true solutions available for 2 or 3 vortices only")
    

    vort_pos=vort_pos_2.copy()
    r_vec= vort_pos[0]-vort_pos[1]
    r=np.linalg.norm(r_vec)
    
    if len_vort==2:
        omega= 1/(np.pi*r*r)
    elif len_vort==3:
        omega= 3.0/(np.pi*r*r*2.0)
    elif len_vort==4:
        omega=2.0/(np.pi*r*r)
            
    center=np.sum(vort_pos, 0)/len_vort
    theta=omega*time
    rotn_mat= [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    temp=np.array([i -center for i in vort_pos])
    temp=np.transpose(temp)
    temp=np.dot(rotn_mat,temp)
    temp=np.transpose(temp)
    true_pos=np.array([i + center for i in temp])
    return(true_pos)  
        




def error_calc(vort_pos_2, gama, npts, n_rotations, euler_or_rk):
    n_iter=100
    time=n_rotations*time_period(vort_pos_2)

    true_position= true_soln(vort_pos_2, time)
    error=np.zeros(npts)
    dt_list=np.zeros(npts)
    for i in prange(npts):
        dt=time/n_iter
        calc_pos, tr= simulation(vort_pos_2, gama, dt, n_iter, euler_or_rk, np.array([[0,0]]))
        n_iter=n_iter+5
        error[i]=np.linalg.norm(true_position[0]-calc_pos[-1][0])
        dt_list[i]=dt
    return(error, dt_list)

@njit        
def net_vel(vort_pos,particle_pos , gama, exception_index=-1):
    '''
    vort_pos: 2d array with each element having [x,y,gama]
    particle_pos:  [x,y]
    exception_index: Index of the vortex which is not contributing towards velocity    
    '''
    vx=0.0
    vy=0.0

    # iter= np.array(prange(len(vort_pos)))
    # if exception_index!=-1:
    #     iter=np.delete(iter,exception_index)

    for i in prange(len(vort_pos)):
        if i!= exception_index:
            dx=particle_pos[0]-vort_pos[i][0]
            dy=particle_pos[1]-vort_pos[i][1]
            rsq=(dx*dx)+(dy*dy)
            vx=vx-(gama[i]*dy)/(2*np.pi*rsq)
            vy=vy+(gama[i]*dx)/(2*np.pi*rsq)

    return(np.array([vx,vy]))
@njit
def velocity_vortices(vort_pos, gama):
    vort_vel=np.zeros_like(vort_pos)
    for i in prange(len(vort_pos)):
        vort_vel[i]=net_vel(vort_pos, vort_pos[i], gama, i)
    return(vort_vel)

@njit
def velocity_tracer(vort_pos, gama, tracer_pos):
    tracer_vel=np.zeros_like(tracer_pos)
    for i in prange((len(tracer_pos))):
        tracer_vel[i]= net_vel(vort_pos,tracer_pos[i], gama)
    return(tracer_vel)
@njit
def euler(v0, s0, dt):
    s1= s0+v0*dt
    return(s1)
@njit
def rk2(v0, v1, vort_pos, dt):
    vort_pos=vort_pos+(dt/2)*(v0+v1)
    return(vort_pos)  

def simulation(vort_pos, gama, dt, iter, euler_or_rk, tracer_pos):
    vortex_all_positions=np.zeros((iter+1,len(vort_pos),2))
    vortex_all_positions[0]=vort_pos.copy()

    tracer_all_positions= np.zeros((iter+1, len(tracer_pos), 2))
    tracer_all_positions[0]= tracer_pos.copy()


    if euler_or_rk==1:
        
        for i in prange(1,iter+1):
            v0_tr=velocity_tracer(vort_pos, gama, tracer_pos)
            tracer_pos=euler(v0_tr, tracer_pos, dt)
            tracer_all_positions[i]= tracer_pos.copy()

            v0= velocity_vortices(vort_pos, gama)
            vort_pos=euler(v0, vort_pos, dt)
            vortex_all_positions[i]=vort_pos.copy()
        return(vortex_all_positions, tracer_all_positions)
        
    elif euler_or_rk==2:
        for i in prange(1,iter+1):
            v0_tr=velocity_tracer(vort_pos, gama, tracer_pos)
            tracer_pos_tmp= euler(v0_tr, tracer_pos, dt)
            v1_tr= velocity_tracer(vort_pos, gama, tracer_pos_tmp)
            tracer_pos= rk2(v0_tr, v1_tr, tracer_pos, dt)
            tracer_all_positions[i]=tracer_pos.copy()

            v0= velocity_vortices(vort_pos, gama)
            vort_pos_tmp=euler(v0, vort_pos, dt)
            v1= velocity_vortices(vort_pos_tmp, gama)
            vort_pos= rk2(v0, v1, vort_pos, dt)
            vortex_all_positions[i]=vort_pos.copy()
        return(vortex_all_positions, tracer_all_positions)



def plot_simulation(vortices_position, gama, T, n_iter, scheme_no, tracer_positions  ,plot_true_positions=0, plot_tracer=0):
    vortex_all_positions, tracer_all_positions =simulation(vortices_position, gama, T/n_iter, n_iter, scheme_no, tracer_positions)

    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.plot(vortex_all_positions[::,::,0], vortex_all_positions[::,::,1])
    if plot_tracer==1:
        plt.plot(tracer_all_positions[::,::,0][-1], tracer_all_positions[::,::,1][-1], "b")

    if plot_true_positions==1:
        vortex_true_position= true_soln(vortices_position, T)
        plt.plot(vortex_true_position[::,0], vortex_true_position[::,1], "*")

    plt.show()


def plot_error(gama):
    error_e, dt_list_e=error_calc(vortices_position, gama, 100, 0.25, 1)
    plt.plot(np.log(dt_list_e), np.log(error_e))

    error_rk, dt_list_rk = error_calc(vortices_position, gama, 100, 0.25, 2)
    plt.plot(np.log(dt_list_rk), np.log(error_rk))

    slope_e=np.polyfit(np.log(dt_list_e), np.log(error_e), 1)[0]

    slope_rk=np.polyfit(np.log(dt_list_rk), np.log(error_rk), 1)[0]

    plt.title("Slope_euler=" +str(slope_e)+ ",  Slope_rk="+ str(slope_rk))
    plt.show()
##############################################################################
##############################################################################


# vortices_position= np.array([[0.5,0.0] , [-0.5,0.0], [0.0, 0.5] ])
# vortex_strength=np.array([1,1, 1])
# tracer_positions= np.array( [[0.0,0.0]])
# T=time_period(vortices_position)/5.0


# plot_simulation(vortices_position, vortex_strength, 50, 5000, 2, tracer_positions  ,plot_true_positions=0, plot_tracer=0)
# plot_error(vortex_strength)


# @njit
# def krasny(w_pos, gamma, delta):
#     len_w= len(w_pos)
#     vel=np.zeros_like(w_pos)
#     # inv=np.array( [ [0.0, -1.0], [1.0, 0.0] ])
#     v=np.zeros(2)
#     tempp=np.zeros(2)
#     for i in prange(len_w):
#         v.fill(0)

#         for k in prange(len_w):
#             dist= w_pos[i]-w_pos[k]
#             r=np.sqrt(dist[0]**2+dist[1]**2)
#             # dist_inv= np.dot(inv, dist)
#             tempp[0]=-dist[1]
#             tempp[1]=dist[0]
#             ker= tempp/(2*np.pi*(r*r+ delta**2))
#             v=v+ker*gamma[k]
            
#         vel[i]=v
#     return(vel)

@njit
def krasny(wpos, gamma, delta):
    '''
        gives velocity at each vortex due to others
    '''
    npt=len(wpos)
    velocities=np.zeros_like(wpos)
    v=np.zeros_like(wpos[0])
    for i in prange(npt):    
        v.fill(0)
        for k in prange(npt):
            dz=wpos[i]-wpos[k]
            ker=gamma[k]/ ( 2*np.pi* (dz[0]**2 + dz[1]**2+ delta**2 ) )
            v[0] = v[0] - dz[1]*ker
            v[1] = v[1] + dz[0]*ker

        velocities[i]=v.copy()
    return(velocities)



def krasny_simulation(vort_pos, gama, delta, dt, iter):
    vortex_all_positions=np.zeros((iter+1,len(vort_pos),2))
    vortex_all_positions[0]=vort_pos.copy()

    for i in prange(1,iter+1):
        v0 = krasny(vort_pos, gama, delta)
        vort_pos_tmp = euler(v0, vort_pos, dt)
        v1 = krasny(vort_pos_tmp, gama, delta)
        vort_pos = rk2(v0, v1, vort_pos, dt)
        vortex_all_positions[i] = vort_pos.copy()
    return(vortex_all_positions)


#complex functions

# @njit
def krasny_vel_at_pt(pt, wpos, gamma, delta, exception_index=-1):
    '''
    returns complex velocity at given point due to krasny blobs
    '''
    v=0+0j
    dz=-wpos+pt
    for k in prange(len(wpos)):
        if k!= exception_index:
            v= v+ (-1j*dz[k].conjugate()*gamma[k] /(2*np.pi*(  (dz[k]* (dz[k].conjugate())) + delta**2 ) ))

    return v.conjugate()


# @njit
def krasny_vel_all_tracer_points(tracer, wpos, gamma, delta):
    v_tracer=np.zeros_like(tracer)
    for i in prange(len(v_tracer)):
        v_tracer[i]=krasny_vel_at_pt(tracer[i], wpos, gamma,delta)#, i)
    return v_tracer

def krasny_vel_all_points_self(tracer, wpos, gamma, delta):
    v_tracer=np.zeros_like(tracer)
    for i in prange(len(v_tracer)):
        v_tracer[i]=krasny_vel_at_pt(tracer[i], wpos, gamma,delta, i)
    return v_tracer

# @njit
def krasny_sim_comp(vort_pos, delta, gamma, dt, iter):
    vort_all_pos=np.zeros( ( iter+1, len(vort_pos) ), np.complex128)
    vort_all_pos[0]=vort_pos
    for i in prange(1,iter+1):
        v0=krasny_vel_all_tracer_points(vort_pos, vort_pos, gamma, delta)
        vort_pos_temp= euler(v0, vort_pos, dt)
        v1=krasny_vel_all_tracer_points(vort_pos_temp, vort_pos_temp, gamma, delta)
        vort_pos=rk2(v0,v1, vort_pos, dt)
        vort_all_pos[i]=vort_pos
    return vort_all_pos

@njit
def vel_chorin(vort_pos, gamma, delta, tracers, self_vel=1):
    v_tracer=np.zeros_like(tracers)
    r=0
    r_mag=0
    v=0+0j
    for i in prange(len(tracers)):
        v=0+0j
        for j in prange(len(vort_pos)):
            if self_vel==0:
                r=tracers[i]-vort_pos[j]
                r_mag=np.abs(r)
                if r_mag>delta:
                    v+= (-1j*r.conjugate()*gamma[j]/(2*np.pi*r_mag*r_mag))
                elif r_mag>0:
                    v+= (-1j*r.conjugate()*gamma[j]/(2*np.pi*delta*r_mag))
            else:
                if i!=j:
                    r=tracers[i]-vort_pos[j]
                    r_mag=np.abs(r)
                    if r_mag>delta:
                        v+= (-1j*r.conjugate()*gamma[j]/(2*np.pi*r_mag*r_mag))
                    else:
                        v+= (-1j*r.conjugate()*gamma[j]/(2*np.pi*delta*r_mag))

        v_tracer[i]=v.conjugate()
    return(v_tracer)
    