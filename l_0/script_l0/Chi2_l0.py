##############
### Para correr el script: python LBS_function_v3.py -data data_file.dat 
############
####### data_file.dat: it is the data file you wish to use for the Chi^2 method
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy import interpolate
import argparse
import os
#####Data path
path = "/Users/atalianb/Documents/data_LBSG/LSBG/LSBG/"
#########
my_parser = argparse.ArgumentParser()#allow_abbrev=False)
my_parser.add_argument('-data',type=str,required=True)
######
args=my_parser.parse_args()
######
data = np.loadtxt(path+args.data)
##########
vecRp_data = np.array([row[0] for row in data])# galactocentric distance [kpc]
vecvRp_data = np.array([row[1] for row in data])# rotation velocity [km/s]
vecerrvRp_data = np.array([row[2] for row in data])# error in rotation velocity [km/s]
#####
##Constant
#######
G_kpc = 4.302e-6#kpc/SolarMass(km/s)^2
####
##Integration values
###
x0_0 = 3.#first integration
x0_0v1 = 0.01# Start of integration. Use 0.01 for continuity in l=3 solution, for 0,1,2 0.0 it's ok
xf_0v1 = 10. # End of integration
step_0 = 0.5
##l=0
u1_0 = 0.63173215# 1st trial value of unknown init. cond.
u2_0 = 1.28125868# 2nd trial value of unknown init. cond.
u_0 = np.array([u1_0, u2_0])
#########
##Integrate function with RK4
########
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
#######
##Shooting method function
#######
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
#########
###Initial conditions array, the first correspond to the initial condition in phi
###For l=0 the inicial condition is always phi=0
def IC(u,k):
    return np.array([k,0.0,u[0],0.0,u[1]])
#####
###ODE System 
####
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
########
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
#######
#The function integrate f0 with the shooting method,
#integrates to find the M(r) and returns the r[kpc] and Vc[km/s] 
#if the last element of r[kpc] array is minor than the last element of the r data array
#takes the last element of the M(r) array and compute the Vc^2 for the rest of the r elements
###########
def Vc_xy(r,m_a,eps,l=0.):
    X0,Y0,root0_f,arr_x0,arr_list0=shooting(f0,u_0,x0_0v1,x0_0,xf_0v1,step_0,k=1.)
    M_r0 = Mass_func(X0,Y0,l=0.)#Integrates rho(r) to obtain M(r)
    Vc2_r0 = Vc2_cir(X0,eps,M_r0)#Vc^2[km/s]^2 theoretical
    X0_units = r_units(X0,eps,m_a)#r[kpc] theoretical
    M_r0_units = M_r0*eps*1.34e-10/m_a#M(r) with Solar Mass units
    if X0_units[-1]<r[-1]:
        #array from last element of the r[kpc] theoretical to the last element of the data array,
        # with 80 elements. It can be replaced by np.arange(X0_units[-1],vecRp_data[-1],0.1) 
        #but you have to be careful in the next function with interpolate
        r_array = np.linspace(X0_units[-1],r[-1],80)
        Vc2_rmayor = G_kpc*M_r0_units[-1]/r_array#Computes Vc^2 with with the last result from M(r)
        Vc2_total = np.append(Vc2_r0,Vc2_rmayor)#creates an array of Vc^2 with Vc2_r0 and Vc2_rmayor
        r_total = np.append(X0_units,r_array)
        return r_total,np.sqrt(Vc2_total)
    else:
        return X0_units,np.sqrt(Vc2_r0)
############
def Vc_xi2(r,m_a,eps,l=0.):
    Vc = Vc_xy(r,m_a,eps)
    #If you want to use np.arange in the previous function, It is recommended to use extrapolate
    f = interpolate.interp1d(Vc[0],Vc[1],fill_value='extrapolate')
    Vc_new = f(r)
    return Vc_new
######
##Temporal function for the Chi^2 
######
def Vc_l0_m_a_eps(r,params):
    m_a,eps = params
    return Vc_xi2(r,m_a,eps,l=0.)
#######
##chi^{2} function
#####
def Xi2_m_a_eps(params):
    m_a,eps = params
    par = np.exp(m_a),np.exp(eps)
    model = Vc_l0_m_a_eps(vecRp_data,par)
    xi  = np.sum((vecvRp_data-model)**2./(vecerrvRp_data)**2.)
    return xi
##########
##Using minimize
#####
x0_m_a_eps = np.array([np.log(1.0e-23),np.log(1.0e-4)])
LS_m_a_eps = optimize.minimize(Xi2_m_a_eps,x0_m_a_eps,method='L-BFGS-B',bounds=((np.log(1.0e-24),np.log(1.0e-20)),(np.log(1.0e-5),np.log(1.)),))
#print(LS_m_a_eps)
#######
##Plot
#####
plt.errorbar(vecRp_data,vecvRp_data,yerr=vecerrvRp_data,fmt='.',label='data')
plt.plot(vecRp_data,Vc_xi2(vecRp_data,np.asscalar(np.exp(LS_m_a_eps.x[0])),np.asscalar(np.exp(LS_m_a_eps.x[1]))),label='fit')
plt.ylabel(r'$v_{c}(r)$[km/s]')
plt.xlabel("r[kpc]")
plt.title(r'$m_{a}$='+str(np.exp(LS_m_a_eps.x[0]))+r'$\epsilon$='+str(np.exp(LS_m_a_eps.x[1])))
plt.xlim(0,vecRp_data[-1]+0.1)
plt.legend(loc='lower right', prop={'size':9})
plt.savefig('Vc_Least_Squares'+str(args.data)+'.pdf')