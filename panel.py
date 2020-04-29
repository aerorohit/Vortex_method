import numpy as np 
import matplotlib.pyplot as plt 
import time
import sys
from numba import njit, prange
import vortex_sim as vsim
import cmath


def plot_all_pos(all_pos):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.plot(all_pos.real, all_pos.imag)
    plt.plot(all_pos[0].real, all_pos[0].imag, "X")
    plot_circle(0+0j, 1)


def plot_circle(cent, r):
    tcir=np.linspace(0,2*np.pi,100)
    plt.plot(r*np.cos(tcir)+cent.real, r*np.sin(tcir)+cent.imag)


def method_of_images(a, pos, gamma, n_iter, dt=0.01):
    '''
    a: radius of cylinder
    pos: initial position of vortex
    gamma: strength of one vortex

    '''
    all_pos= np.zeros(n_iter+1, dtype="complex")
    all_pos[0]=pos
    for i in prange(1,n_iter+1):
        im = (a*a)/pos.conjugate()
        v0=vsim.krasny_vel_at_pt(pos, np.array([im, 0+0j]), np.array([-1, 1])*gamma, 0) 
        s1=vsim.euler(v0, pos,dt)
        im2=(a*a)/(s1.conjugate())
        v1= vsim.krasny_vel_at_pt(pos, np.array([im2, 0+0j]), np.array([-1, 1])*gamma, 0)
        pos=vsim.rk2(v0,v1, pos, dt)
        all_pos[i]=pos

    plot_all_pos(all_pos)
    return all_pos


def problem2():
    

    panels=get_panels_circle(1,0,0,40)
    cpt=control_pt(panels, 0.5)

    pos= -1.5+0j
    dt=0.01
    n_iter=100
    all_pos=np.zeros(n_iter+1, dtype="complex")
    all_pos[0]=pos

    
    for i in prange(1, n_iter+1):

        vel=get_vel_at_boundary(cpt, 0, 0, np.array([pos]), np.array([2*np.pi]))
        A, B= get_gamma_panel(cpt, panels, vel)
        gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]
        
        # contour_panel(5,5,panels, gamma_panel)
        # plt.show()
        v0=vsim.krasny_vel_at_pt(pos, cpt, gamma_panel, 0)
        s1=vsim.euler(v0, pos, dt)

        

        vel=get_vel_at_boundary(cpt, 0, 0, np.array([s1]), np.array([2*np.pi]))
        A, B= get_gamma_panel(cpt, panels, vel)
        gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]

        v1= vsim.krasny_vel_at_pt(s1, cpt, gamma_panel, 0)
        pos = vsim.rk2(v0, v1, pos, dt)

        all_pos[i]=pos

    plot_all_pos(all_pos)
    return all_pos


def problem2_with_constant_panels():
    panels=get_panels_circle(1,0,0,40)
    cpt=control_pt(panels, 0.5)

    pos= -1.5+0j
    dt=0.01
    n_iter=300
    all_pos=np.zeros(n_iter+1, dtype="complex")
    all_pos[0]=pos

    for i in prange(1, n_iter+1):

        vel=get_vel_at_boundary(cpt, 0, 0, np.array([pos]), np.array([2*np.pi]))
        gamma_panel= get_gamma_constant_panels(cpt, panels, vel)
        
        # gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]
        
        v0=vel_due_to_constant_panel(panels, gamma_panel, np.array([pos]))[0]
        s1=vsim.euler(v0, pos, dt)

        vel=get_vel_at_boundary(cpt, 0, 0, np.array([s1]), np.array([2*np.pi]))
        gamma_panel= get_gamma_constant_panels(cpt, panels, vel)
        # gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]

        v1= vel_due_to_constant_panel(panels, gamma_panel, np.array([s1]))[0]
        pos = vsim.rk2(v0, v1, pos, dt)

        all_pos[i]=pos

    plot_all_pos(all_pos)
    return all_pos


def problem2_with_linear_panels():
    

    panels=get_panels_circle(1,0,0,30)
    cpt=control_pt(panels, 0.5)

    pos= -1.5+0j
    dt=0.01
    n_iter=300
    all_pos=np.zeros(n_iter+1, dtype="complex")
    all_pos[0]=pos

    for i in prange(1, n_iter+1):

        vel=get_vel_at_boundary(cpt, 0, 0, np.array([pos]), np.array([2*np.pi]))
        gamma_panel= get_gamma_linear_panels(cpt, panels, vel)
        # gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]
        v0=vel_due_to_panel(panels, gamma_panel, np.array([pos]))[0]
        s1=vsim.euler(v0, pos, dt)

        vel=get_vel_at_boundary(cpt, 0, 0, np.array([s1]), np.array([2*np.pi]))
        gamma_panel= get_gamma_linear_panels(cpt, panels, vel)
        # gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]

        v1= vel_due_to_panel(panels, gamma_panel, np.array([s1]))[0]
        pos = vsim.rk2(v0, v1, pos, dt)

        all_pos[i]=pos

    plot_all_pos(all_pos)
    return all_pos


def true_solution(a,v_inf, points):
    true_vel=np.zeros_like(points)  
    for i in prange(len(points)):
        pol= cmath.polar(points[i])
        temp =[v_inf*( 1- (a**2/pol[0]**2) )*np.cos(pol[1]), v_inf*( 1+ (a**2/pol[0]**2) )*np.sin(pol[1])]
        rotn=np.array([  [np.cos(pol[1]), np.sin(pol[1])] , [-np.sin(pol[1]), np.cos(pol[1]) ]  ])
        v=np.dot(rotn, temp)
        true_vel[i]=complex(v[0], v[1])
    return true_vel.conjugate()

@njit
def get_panels_circle(r,x,y,n):
    '''
    Anti-clockwise arranged
    '''
    pan=np.zeros(n, np.complex128)
    theta=np.linspace(0, 2*np.pi,n+1)[:-1]
    for i in prange(n):
        pan[i]=complex(r*np.cos(theta[i])+x, r*np.sin(theta[i])+y)

    return pan

@njit
def control_pt(panels, t):
    '''
        t: fraction of distance at which we want control point to be placed
    '''
    cpt=np.zeros_like(panels)
    for i in prange(len(panels)-1):
        cpt[i]=t*panels[i]+(1-t)*panels[i+1]

    cpt[-1]=t*panels[-1]+(1-t)*panels[0]
    return cpt
    
@njit
def get_normals(panels):
    '''
    considers closed body, travels in anticlockwise direction
    '''
    if len(panels)<3:
        # sys.exit("provide more than 2 points to get closed figure")
        return
    else:
        norm=np.zeros_like(panels)
        for i in prange(0,len(panels)-1):
            norm[i]=(panels[i]-panels[i+1])*1j
            norm[i]=norm[i]/np.abs(norm[i])

        norm[-1]=(panels[-1]-panels[0])*1j
        norm[-1]=norm[-1]/np.abs(norm[-1])
        return(norm)

@njit
def get_vel_at_boundary(panels, v_inf, vb, wpos=[],  gamma_vort=[], delta=0.00):
    vel=np.zeros_like(panels)
    if len(wpos)!=0:
        return( vsim.krasny_vel_all_tracer_points(panels, wpos, gamma_vort, delta) + v_inf - vb  )
    else:
        vel=vel + v_inf - vb
        return(vel)

@njit
def get_gamma_panel(cpt, panels, vel ):
    '''
        panels are approximated by point vortices

        cpt: control points
        panel: panel coordinates
        velocity should be calculated at control point
        This function considers point vortices
    '''
    n=len(cpt)
    normals=get_normals(panels)
    B=np.zeros(n+1)
    A=np.zeros(( n+1, n))
    for i in prange(n):
        B[i]=-(vel[i].real*normals[i].real+vel[i].imag*normals[i].imag)
        for j in prange(n):
            if i!=j:
                A[i][j]=(-1j/(2*np.pi*(cpt[i]-cpt[j]))).real*normals[i].real - (-1j/(2*np.pi*(cpt[i]-cpt[j]))).imag*normals[i].imag
    A[n]=np.ones(n)

    return A,B
        
@njit
def get_gamma_linear_panels(cpt, panels, vel):
    '''
        Calculates A & B matrix for panels having linearly varying circulation
    '''
    n=len(cpt)
    normals=get_normals(panels)
    l=np.abs(panels[0]-panels[1])
    B=np.zeros(n+1)
    A=np.zeros(( n+1, n))
    A2=np.zeros(( n+1, n))
    zprime=0+0j
    vz1=0+0j
    vz2=0+0j
    r=0
    theta=0
    for i in prange(n):
        B[i]= -(vel[i].real*normals[i].real+vel[i].imag*normals[i].imag)
        for j in prange(n):
            if j==(n-1):
                theta=cmath.phase(panels[0]-panels[n-1])
            else:   
                theta=cmath.phase(panels[j+1]-panels[j])
            zprime=(cpt[i] -panels[j])*np.e**(-1j*theta)
            vz1= (((-1j/(2*np.pi))* ( (((zprime/l)-1)*np.log(1-l/zprime)) +1  )).conjugate()) *np.e**(1j*theta)
            vz2= (((1j/(2*np.pi))*( (zprime/l)*np.log(1-l/zprime) + 1)).conjugate()) *np.e**(1j*theta)
            A[i][j] +=  vz1.real*normals[i].real+vz1.imag*normals[i].imag
            if j==(n-1):
                A[i][0]+= vz2.real*normals[i].real+vz2.imag*normals[i].imag
            else:
                A[i][j+1]+= vz2.real*normals[i].real+vz2.imag*normals[i].imag
    

    A[n]=np.ones(n)
    gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]
    return gamma_panel

@njit
def get_gamma_constant_panels(cpt, panels, vel):
    n=len(cpt)
    normals=get_normals(panels)
    l=np.abs(panels[0]-panels[1])
    B=np.zeros(n+1)
    A=np.zeros(( n+1, n))
    zprime=0+0j
    vz=0+0j
    r=0
    theta=0
    for i in prange(n):
        B[i]=-(vel[i].real*normals[i].real+vel[i].imag*normals[i].imag)
        for j in prange(n):
            if i!=j:
                if j==(n-1):
                    theta=cmath.phase(panels[0]-panels[n-1])
                else:   
                    theta=cmath.phase(panels[j+1]-panels[j])

                zprime=(cpt[i] -panels[j])*np.e**(-1j*theta)
                vz= (((1j/(2*np.pi))* ( np.log(1-l/zprime)  )).conjugate()) *np.e**(1j*theta)
                A[i][j]=  vz.real*normals[i].real+vz.imag*normals[i].imag
    A[n]=np.ones(n)
    gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]
    return gamma_panel

@njit
def vel_due_to_panel(panels, gamma, points):
    '''
    considers closed loop of panels having linearly varying circulation

    '''
    r=0
    theta=0
    zprime=0+0j
    vz1=0+0j
    vz2=0+0j
    l=np.abs(panels[0]-panels[1])
    v=0+0j
    v_all=np.zeros_like(points)
    n=len(panels)

    for i in prange(len(points)):
        v=0+0j
        for j in prange(len(panels)-1):
            
            theta=cmath.phase(panels[j+1]-panels[j])
            
            l=np.abs(panels[j]-panels[j+1])
            zprime=(points[i] -panels[j])*np.e**(-1j*theta)
            vz2= (1j*gamma[j+1]/(2*np.pi))* (( (zprime/l)*np.log(1-l/zprime) ) + 1)
            vz1= (-1j*gamma[j]/(2*np.pi))* ( ( ((zprime/l)-1)*np.log(1-l/zprime) ) +1  )
            v= v + ((vz1 + vz2).conjugate())*(np.e**(1j*theta))

        theta=cmath.phase(panels[0]-panels[n-1])
        zprime=(points[i] -panels[n-1])*np.e**(-1j*theta)
        l=np.abs(panels[n-1]-panels[0])
        vz1= (-1j*gamma[n-1]/(2*np.pi))* ( ((zprime/l)-1)*np.log(1-l/zprime)+1  )
        vz2= (1j*gamma[0]/(2*np.pi))*( (zprime/l)*np.log(1-l/zprime) + 1)
        v= v + ((vz1 + vz2).conjugate())*(np.e**(1j*theta))

        v_all[i]=v
    
    return v_all
    
@njit   
def vel_due_to_constant_panel(panels, gamma, points):
    '''
    considers closed loop of panels
    '''
    r=0
    theta=0
    zprime=0+0j
    vz1=0+0j
    l=np.abs(panels[0]-panels[1])
    v=0+0j
    v_all=np.zeros_like(points)
    n=len(panels)

    for i in prange(len(points)):
        v=0+0j
        for j in prange(len(panels)):
            if j==(n-1):
                theta=cmath.phase(panels[0]-panels[n-1])
                l=np.abs(panels[j]-panels[0])
            else:   
                theta=cmath.phase(panels[j+1]-panels[j])
                l=np.abs(panels[j]-panels[j+1])

            zprime=(points[i] -panels[j])*np.e**(-1j*theta)
            
            vz1= (1j*gamma[j]/(2*np.pi))* ( ( np.log(1-l/zprime) )  )

            v= v + ((vz1).conjugate())*(np.e**(1j*theta))
        v_all[i]=v
    
    return v_all



def contour_num(dx, n):
    r=np.linspace(1.01, 1+dx, n)
    t=np.linspace(0,np.pi*2, 31)
    r, t=np.meshgrid(r,t)
    x=r*np.cos(t)
    y=r*np.sin(t)
    nx=x.shape[0]
    mat=np.hstack(x)+1j*np.hstack(y)
    ts=true_solution(1, 1, mat)
    ts=ts.reshape((31, n))
    plt.figure()
    plt.axes().set_aspect("equal")
    plt.title("True Potential Solution")
    plt.contourf(x, y, np.abs(ts))
    plt.colorbar()
    tcir=np.linspace(0,2*np.pi,100)
    plt.plot(np.cos(tcir), np.sin(tcir))
    plt.quiver(x, y, ts.real, ts.imag)

def contour_panel(dx, n, panels, gamma_panel, constant_or_linear=0):
    '''
    constant_or_linear: 0->constant
    '''
    # r=np.linspace(.1, 1.1+dx, n)
    # t=np.linspace(0,np.pi*2, 31)
    # r, t=np.meshgrid(r,t)
    # x=r*np.cos(t) 
    # y=r*np.sin(t) 
    x=np.linspace(-3, 3, 21)
    y=np.linspace(-2, 2, 21)
    x, y= np.meshgrid(x,y)
    nx=x.shape[0]
    mat=np.hstack(x)+1j*np.hstack(y)
    if constant_or_linear==1:
        ts=vel_due_to_panel(panels, gamma_panel, mat)+(1+0j)
        title="Solution using linearly varying panels"
    if constant_or_linear==0:
        ts=vel_due_to_constant_panel(panels, gamma_panel, mat)+(1+0j)
        title= "solution using Constant intensity panels"
    
    ts=ts.reshape((21, 21))
    plt.figure()
    plt.axes().set_aspect("equal")
    plt.title(title)
    # plt.contourf(x,y, np.abs(ts))
    plot_circle(0+0j,1)
    plt.quiver(x, y, ts.real, ts.imag)
    # plt.colorbar()



def problem1(n_panel):
    panels=get_panels_circle(1,0,0,n_panel)
    cpt=control_pt(panels, 0.5)
    # print("panel point positions center")
    # print(cpt)
    vel=get_vel_at_boundary(cpt,1+0j, 0+0j)

    A,B=get_gamma_panel(cpt, panels, vel)
    # print("A\n", A, "\nB\n", B)
    gamma_panel= np.linalg.lstsq(A,B, rcond=-1)[0]
    # print("gamma")
    # print(gamma_panel)
    print("gamma sum ", np.sum(gamma_panel))

    # vort_center= -np.sum(gamma_panel)

    tracers=get_panels_circle(2,0,0,1)
    # print("tracers")
    # print(tracers)

    vel_tracer= get_vel_at_boundary(tracers, 1+0j, 0+0j, cpt, gamma_panel, 0.0 )
    
    vel_tracers_true= true_solution(1,1, tracers)



    dx= ( vel_tracer-vel_tracers_true )* 100/np.abs(vel_tracers_true)
    # print("vel at  tracer")
    # print(vel_tracer)
    print("percentage error", np.average(np.abs(dx)))
    # print(np.average(np.abs(vel_tracers_true)))
    contour_calc(1, 5, cpt, gamma_panel)
    contour_num(1,5)
    # plt.show()
    return(np.average(np.abs(dx)))
    

def problem1_panels(n_panel, r , constant_or_linear=0, plot_bool=0):
    panels=get_panels_circle(1,0,0,n_panel)
    cpt=control_pt(panels, 0.5)
    vel=np.array([1+0j]*len(cpt))
    tracers=get_panels_circle(r,0,0,200)
    if constant_or_linear==0:
        gamma_panel= get_gamma_constant_panels(cpt, panels, vel)
        v1= vel_due_to_constant_panel(panels, gamma_panel, tracers)+(1+0j)
        title="constant_panels"
    elif constant_or_linear==1:
        gamma_panel=get_gamma_linear_panels(cpt, panels, vel)
        v1= vel_due_to_panel(panels, gamma_panel, tracers)+(1+0j)
        title="linear panels"

    v2 =true_solution(1,1, tracers)
    dx= (v1-v2)*100/np.abs(v2)
    err=np.average(np.abs(dx))
    if plot_bool!=0:
        print(n_panel,"percentage error "+ title +": ", err)
    if plot_bool!=0:
        if constant_or_linear==0:
            contour_panel(1, 25, panels, gamma_panel, 0)
        elif constant_or_linear==1:
            contour_panel(1, 25, panels, gamma_panel, 1)
    return(err)


def vel_one_panel_troubleshoot(points, theta=0):
    v=np.zeros_like(points)
    points=points*(np.e**(-1j*theta))
    for i in prange(len(points)):
        v[i]=((1j/(2*np.pi))*np.log(1-1/points[i])).conjugate()
    return v*(np.e**(1j*theta))


#############################################

if __name__ == "__main__":

    
    # problem1_panels(100, 1, 1)
    # problem1_panels(100, 1)
   
    npan=np.array(prange(20, 101))
    # npan=np.delete(npan, 58)
    err=np.zeros(len(npan))
    for i in prange(len(npan)):
        err[i]=problem1_panels(npan[i], 2, 0 )
    
    plt.plot(npan, err)
    # print(np.argmax(err))
    plt.show()

