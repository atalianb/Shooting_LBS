###########
##To run the script: python phi0_combinations.py -data data_file.dat
#################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import interpolate
import argparse
import time

time_i = time.time()
######
##if you wish to save the results change to True
################
store = True
#######
my_parser = argparse.ArgumentParser()#allow_abbrev=False)
my_parser.add_argument('-data',type=str,required=True)
#####
args=my_parser.parse_args()
print(args.data)
######
#####Data path
data_path = "/Users/atalianb/Documents/data_LBSG/Blok_McGaugh_& _Rubin_(2001)/"
data = np.loadtxt(data_path+args.data)
##########
vecRp_data = np.array([row[0] for row in data])# galactocentric distance [kpc]
vecvRp_data = np.array([row[1] for row in data])# rotation velocity [km/s]
vecerrvRp_data = np.array([row[2] for row in data])# error in rotation velocity [km/s]
####
#####Gravitational Constant
G_kpc = 4.302e-6#kpc/SolarMass(km/s)^2
####
##Integration values
###
x0_0 = 3.#first integration
x0_0v1 = 0.01# Start of integration. Use 0.01 for continuity in l=3 solution, for 0,1,2 0.0 it's ok
xf_0v1 = 10.# End of integration
step_0 = 0.5#Step to integrate from x0_0 until xf_0v1 is reached
##l=0
u1_0 = 0.63173215# 1st trial value of unknown init. cond.
u2_0 = 1.28125868# 2nd trial value of unknown init. cond.
u_0 = np.array([u1_0, u2_0])
###l=1 
u1_1 = -3.7# 1st trial value of unknown init. cond.
u2_1 = -2.2# 2nd trial value of unknown init. cond.
u_1 = np.array([u1_1, u2_1])
#####
def Integrate(func,x0,y0,x,h):
    # Finds value of y for a given x using step size h 
    # and initial value y0 at x0.
    def RK4(func,x0,y0,h):
        K0 = h*func(x0,y0)
        K1 = h*func(x0 + 0.5*h, y0 + 0.5*K0)
        K2 = h*func(x0 + 0.5*h, y0 + 0.5*K1)
        K3 = h*func(x0 + h, y0 + K2)
        return (K0 + 2.*K1 + 2.*K2 + K3)/6.
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
         # Count number of iterations using step size or
        h = min(h,x - x0)# step height h
        y0 = y0 + RK4(func,x0,y0,h)#update next value of y
        x0 = x0 + h#update next value of x
        X.append(x0)
        Y.append(y0)
    return np.array(X),np.array(Y)
#####
def shooting(func,u,x0,x,xf,step,k,h=0.01):
    def res(u):# Boundary condition residual
        X,Y = Integrate(func,x0,IC(u,k),x,h)
        y = Y[len(Y) - 1]#last value of Y
        r = np.zeros(len(u))
        r[0] = y[0]#y0(inf)=0
        r[1] = y[2]/x + y[3]#y_3(inf)/r + y_4(inf)=0
        return r
    x_list = []
    root_list = []
    while x<=xf:
        x_list.append(x)
        root = optimize.root(res,u)
        u = root.x
        root_temp = optimize.root(res,root.x)
        root_list.append(root_temp.x)
        X,Y = Integrate(func,x0,IC(root_temp.x,k),x,h)
        x = x+step
    return X,Y,root_temp,np.array(x_list),np.array(root_list)
#######
def IC(u,k):#Initial conditions array, the first correspond to the initial condition in phi
    return np.array([k,0.0,u[0],0.0,u[1]])
#########
##Diferentian equations system for ell = 0 and ell = 1
def f0(x,y):
    l = 0.
    F = np.zeros(5)
    if x==0:
        F[0] = y[1]
        F[1] = 2.*y[0]*(y[2]-y[4])
        F[2] = y[3]
        F[3] = 0.
        F[4] = 0.
    else:
        F[0] = y[1]
        F[1] = -2.*(l+1.)*y[1]/x +2.*y[0]*(y[2]-y[4])
        F[2] = y[3]
        F[3] = (2.*l+1.)*x**(2.*l)*y[0]**2. - 2.*y[3]/x
        F[4] = 0.
    return F
def f1(x,y):
    l = 1.
    F = np.zeros(5)
    if x==0:
        F[0] = y[1]
        F[1] = 2.*y[0]*(y[2]-y[4])
        F[2] = y[3]
        F[3] = 0.
        F[4] = 0.
    else:
        F[0] = y[1]
        F[1] = -2.*(l+1.)*y[1]/x +2.*y[0]*(y[2]-y[4])
        F[2] = y[3]
        F[3] = (2.*l+1.)*x**(2.*l)*y[0]**2. - 2.*y[3]/x
        F[4] = 0.
    return F
#####
##Integrate the density (rho(r)) to obtain the mass function
#####
def Mass_func(r,phi,l):
    Int = np.zeros(len(r))
    dr = np.diff(r)[0]
    phi_array = np.array(phi[:,0])
    for i in range(0,len(r)-1):
        Int[i+1] = dr*(phi_array[i+1]**2.*r[i+1]**(2.*l+2.)) + Int[i]
    return Int
#########################################
###Circular Speed function in (km/s)^2###
#########################################
def Vc2_cir(r,eps,M):
    units =8.95e10*eps**2.
    return (units*M)/r
####
##Units for r in kpc
###
def r_units(r,eps,m_a):
    return (6.39e-27*r)/(eps*m_a)
###############################
###Rescaling function for x axis, in y just multiply by phi_0
####################
def rescaling(phi_0,l):
    lmbd = (1./phi_0)**(1./(l+2.))
    return lmbd
######################
#The function integrate f for the l of your choise with the shooting method,
#taking s_begin as the initial array for the shooting method.
#integrates to find the M(r) and returns the r[kpc] and Vc[km/s] 
#if the last element of r[kpc] array is minor than the last element of the r data array
#takes the last element of the M(r) array and compute the Vc^2 for the rest of the r elements
def Vc_xy_phi(r,m_a,eps,phi0,func,s_begin,l):
    X0,Y0,root0_f,arr_x0,arr_list0=shooting(func,s_begin,x0_0v1,x0_0,xf_0v1,step_0,k=phi0)
    Xl = X0*rescaling(phi0,l)
    M_r0 = Mass_func(Xl,Y0*phi0,l)#Integrates rho(r) to obtain M(r)
    Vc2_r0 = Vc2_cir(Xl,eps,M_r0)#Vc^2[km/s]^2 theoretical
    X0_units = r_units(Xl,eps,m_a)#r[kpc] theoretical
    M_r0_units = M_r0*eps*1.34e-10/m_a#M(r) with Solar Mass units
    if X0_units[-1]<r[-1]:
        #array from last element of the r[kpc] theoretical to the last element of the data array,
        # with 80 elements. It can be replaced by np.arange(X0_units[-1],vecRp_data[-1],0.1) 
        #but you have to be careful in the next function with interpolate
        r_array = np.linspace(X0_units[-1],r[-1],80)
        Vc2_rmayor = G_kpc*M_r0_units[-1]/r_array#Computes Vc^2 with with the last result from M(r)
        Vc2_total = np.append(Vc2_r0,Vc2_rmayor)#creates an array of Vc^2 with Vc2_r0 and Vc2_rmayor
        r_total = np.append(X0_units,r_array)
        return r_total,Vc2_total
    else:
        return X0_units,Vc2_r0
#######
def Vc_xi2_phi(r,m_a,eps,phi0,func,u_begin,l):
    Vc = Vc_xy_phi(r,m_a,eps,phi0,func,u_begin,l)
    #If you want to use np.arange in the previous function, It is recommended to use extrapolate
    f = interpolate.interp1d(Vc[0],Vc[1],fill_value='extrapolate')
    Vc_new = f(r)
    return Vc_new
########
def Vc_m_a_eps_phi0_01(r,params,phi0_0,phi0_1):
    m_a,eps0,eps1 = params
    Vc2 = Vc_xi2_phi(r,m_a,eps0,phi0_0,f0,u_0,l=0.) + Vc_xi2_phi(r,m_a,eps1,phi0_1,f1,u_1,l=1.)
    return np.sqrt(Vc2)
#######
def Xi2_m_a_eps_phi01_data(params,phi00,phi01):
    m_a,eps0,eps1 = params
    par = np.exp(m_a),np.exp(eps0),np.exp(eps1)
    model = Vc_m_a_eps_phi0_01(vecRp_data,par,phi00,phi01)
    xi  = np.sum((vecvRp_data-model)**2./(vecerrvRp_data)**2.)
    return xi
####
#array of initial contiditions to phi
#####
phi0_array = np.arange(0.001,1.,0.02)
#####
chi2_vals = np.zeros((len(phi0_array),len(phi0_array),3))
status_list = []
fun_list = []
for i in range(len(phi0_array)):
    for j in range(len(phi0_array)):
        x0_m_a_eps_phi0_data = np.array([np.log(1.0e-23),np.log(1.0e-3),np.log(1.0e-3)])
        LS_m_a_eps_phi0_data = optimize.minimize(Xi2_m_a_eps_phi01_data,x0_m_a_eps_phi0_data,method='L-BFGS-B',bounds=((np.log(1.0e-24),np.log(1.0e-22)),(np.log(1.0e-4),np.log(1.0e-2)),(np.log(1.0e-4),np.log(1.0e-2)),),args=(phi0_array[i],phi0_array[j]))
        chi2_vals[i,j,:] = LS_m_a_eps_phi0_data.x
        status_list.append(LS_m_a_eps_phi0_data.success)
        fun_list.append(LS_m_a_eps_phi0_data.fun)
########
####
if store==True:
    store_path = "phi0_l0_l1_combinations/"
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    np.savetxt(store_path+'Chi2'+str(args.data)+'.dat',np.exp(chi2_vals.reshape((3,-1))),header=str(chi2_vals.shape))
    status_fun_array = np.array([status_list,fun_list])
    np.savetxt('status_and_fun_values'+str(args.data)+'.dat',status_fun_array)
