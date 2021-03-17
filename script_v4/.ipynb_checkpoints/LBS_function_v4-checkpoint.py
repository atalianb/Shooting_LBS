##############
### Para correr el script: python LBS_function_v3.py -data data_file.dat -phi0_0 number -phi0_1 number -phi0_2 number -phi0_3 number
##############
####### data_file.dat: it is the data file you wish to plot
####### parameter -phi0_0: the initial condition for phi with l=0, is a number between 0.1 and 1
####### parameter -phi0_1: the initial condition for phi with l=1, is a number between 0.1 and 1
####### parameter -phi0_2: the initial condition for phi with l=2, is a number between 0.1 and 1
####### parameter -phi0_3: the initial condition for phi with l=3, is a number between 0.1 and 1
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
import argparse
import time
import os

time_i = time.time()
#######
##if you wish to save the results change to True
################
store = True
#####Data path
path = "/Users/atalianb/Documents/data_LBSG/LSBG/LSBG/"
#########
my_parser = argparse.ArgumentParser()#allow_abbrev=False)
my_parser.add_argument('-data',type=str,required=True)
my_parser.add_argument('-phi0_0',action='store',type=float,required=True)
my_parser.add_argument('-phi0_1',action='store',type=float,required=True)
my_parser.add_argument('-phi0_2',action='store',type=float,required=True)
my_parser.add_argument('-phi0_3',action='store',type=float,required=True)
#####
args=my_parser.parse_args()
print(args.phi0_0,args.phi0_1,args.phi0_2)
print(args.data)
######
data = np.loadtxt(path+args.data)
##########
vecRp_data = np.array([row[0] for row in data])# galactocentric distance [kpc]
vecvRp_data = np.array([row[1] for row in data])# rotation velocity [km/s]
vecerrvRp_data = np.array([row[2] for row in data])# error in rotation velocity [km/s]
####
##Values
####
eps = 2.5e-4#
m_a = 1.0e-22#eV/c^2
#r_num = vecRp_data[-1]*eps*m_a/6.39e-27
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
###l=1 
u1_1 = -3.7# 1st trial value of unknown init. cond.
u2_1 = -2.2# 2nd trial value of unknown init. cond.
u_1 = np.array([u1_1, u2_1])
##l=2
u1_2 = -3.75086524# 1st trial value of unknown init. cond.
u2_2 = -2.21111819# 2nd trial value of unknown init. cond.
u_2 = np.array([u1_2, u2_2])
##l=3
u1_3 = -10.99374281# 1st trial value of unknown init. cond.
u2_3 = -6.46898492# 2nd trial value of unknown init. cond.
u_3 = np.array([u1_3, u2_3])
####
###
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
#########
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
        #print("x=",x)
        x_list.append(x)
        #print("u=",u)
        root = optimize.root(res,u)
        #print("root=",root)
        u = root.x
        #print("u=",u)
        root_temp = optimize.root(res,root.x)
        #print("root_temp=",root_temp)
        root_list.append(root_temp.x)
        X,Y = Integrate(func,x0,IC(root_temp.x,k),x,h)
        x = x+step
    return X,Y,root_temp,np.array(x_list),np.array(root_list)
########
def IC(u,k):#Initial conditions array, the first correspond to the initial condition in phi
    return np.array([k,0.0,u[0],0.0,u[1]])
###
###
###ODE system
####
###
########
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
##########
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
############
def f2(x,y):
    l = 2.
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
############
def f3(x,y):
    l = 3.
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
def Vc2_cir(M,r):
    units =8.95e10*eps**2.
    return (units*M)/r
####
##Units for r in kpc
###
def r_units(r):
    return (6.39e-27*r)/(eps*m_a)
###############################
###Rescaling function for x axis, in y just multiply by phi_0
####################
def rescaling(phi_0,l):
    lmbd = (1./phi_0)**(1./(l+2.))
    return lmbd
#####################
###Computing
#################
X0_f,Y0_f,root0_f,arr_x0,arr_list0=shooting(f0,u_0,x0_0v1,x0_0,xf_0v1,step_0,k=1.)
X1_f,Y1_f,root1_f,arr_x1,arr_list1=shooting(f1,u_1,x0_0v1,x0_0,xf_0v1,step_0,k=1.)
X2_f,Y2_f,root2_f,arr_x2,arr_list2=shooting(f2,u_2,x0_0v1,x0_0,xf_0v1,step_0,k=1.)
X3_f,Y3_f,root3_f,arr_x3,arr_list3=shooting(f3,u_3,x0_0v1,x0_0,xf_0v1,step_0,k=1.)
###
###
###
Xl0 = X0_f*rescaling(args.phi0_0,l=0.)
Xl1 = X1_f*rescaling(args.phi0_1,l=1.)
Xl2 = X2_f*rescaling(args.phi0_2,l=2.)
Xl3 = X3_f*rescaling(args.phi0_3,l=3.)
############
##Mass
###########
M_r0 = Mass_func(Xl0,Y0_f*args.phi0_0,l=0.)
M_r1 = Mass_func(Xl1,Y1_f*args.phi0_1,l=1.)
M_r2 = Mass_func(Xl2,Y2_f*args.phi0_2,l=2.)
M_r3 = Mass_func(Xl3,Y3_f*args.phi0_3,l=3.)
################################
##Computing the circular speed
#################################
Vc2_r0 = Vc2_cir(M_r0,X0_f)
Vc2_r1 = Vc2_cir(M_r1,X1_f)
Vc2_r2 = Vc2_cir(M_r2,X2_f)
Vc2_r3 = Vc2_cir(M_r3,X3_f)
#####################
##Computing r in kpc
########################
X0_units = r_units(Xl0)
X1_units = r_units(Xl1)
X2_units = r_units(Xl2)
X3_units = r_units(Xl3)
######################################
##Sum the l=0, l=1 and l=2 Circular speed
##########################################
Vc_tot = np.sqrt(Vc2_r0 + Vc2_r1 + Vc2_r2 + Vc2_r3)
################
###Plot
#############
plt.errorbar(vecRp_data,vecvRp_data,yerr=vecerrvRp_data,fmt='.',label='data')
plt.plot(X0_units,np.sqrt(Vc2_r0),label='l=0'r'$ \phi_{0}=$'+str(args.phi0_0))
plt.plot(X1_units,np.sqrt(Vc2_r1),label='l=1'r'$ \phi_{0}=$'+str(args.phi0_1))
plt.plot(X2_units,np.sqrt(Vc2_r2),label='l=2'r'$ \phi_{0}=$'+str(args.phi0_2))
plt.plot(X3_units,np.sqrt(Vc2_r3),label='l=3'r'$ \phi_{0}=$'+str(args.phi0_3))
plt.plot(X0_units,Vc_tot,label='total')
plt.ylabel(r'$v_{c}(r)$[km/s]')
plt.xlabel("r[kpc]")
plt.title(args.data)
plt.legend(loc='upper left', prop={'size':9})
plt.title(str(args.data))
plt.xlim(0,vecRp_data[-1]+0.1)
plt.savefig('Vc_tot_l0123_phi0_'+str(args.phi0_0)+'phi1_'+str(args.phi0_1)+'phi2_'+str(args.phi0_2)+'phi3_'+str(args.phi0_3)+'.pdf')
###########
####
if store==True:
    store_path = "files_LBS/"
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    #####l=0
    X0_Y0_array = np.array([X0_f,Y0_f[:,0]])
    np.savetxt(store_path+'XY_l0_phi'+str(args.phi0_0)+'.dat',X0_Y0_array.T)
    rootslist0 = np.array([arr_x0,arr_list0[:,0],arr_list0[:,1]])
    np.savetxt(store_path+'roots0_phi'+str(args.phi0_0)+'.dat',rootslist0.T)
    M_data0 = np.array([X0_f,M_r0])
    np.savetxt('M_l0_phi'+str(args.phi0_0)+'.dat',M_data0.T)
    Vc2_data0 = np.array([X0_units,Vc2_r0])
    np.savetxt(store_path+'Vc2_l0_phi'+str(args.phi0_0)+'.dat',Vc2_data0.T)
    ######l=1
    X1_Y1_array = np.array([X1_f,Y1_f[:,0]])
    np.savetxt(store_path+'XY_l1_phi'+str(args.phi0_1)+'.dat',X1_Y1_array.T)
    rootslist1 = np.array([arr_x1,arr_list1[:,0],arr_list1[:,1]])
    np.savetxt(store_path+'roots1_phi'+str(args.phi0_1)+'.dat',rootslist1.T)
    M_data1 = np.array([X1_f,M_r1])
    np.savetxt(store_path+'M_l1_phi'+str(args.phi0_1)+'.dat',M_data1.T)
    Vc2_data1 = np.array([X1_units,Vc2_r1])
    np.savetxt(store_path+'Vc2_l1_phi'+str(args.phi0_1)+'.dat',Vc2_data1.T)
    ######l=2
    X2_Y2_array = np.array([X2_f,Y2_f[:,0]])
    np.savetxt(store_path+'XY_l2_phi'+str(args.phi0_2)+'.dat',X2_Y2_array.T)
    rootslist2 = np.array([arr_x2,arr_list2[:,0],arr_list2[:,1]])
    np.savetxt(store_path+'roots2_phi'+str(args.phi0_2)+'.dat',rootslist2.T)
    M_data2 = np.array([X2_f,M_r2])
    np.savetxt(store_path+'M_l2_phi'+str(args.phi0_2)+'.dat',M_data2.T)
    Vc2_data2 = np.array([X2_units,Vc2_r2])
    np.savetxt(store_path+'Vc2_l2_phi'+str(args.phi0_2)+'.dat',Vc2_data2.T)
    ######l=3
    X3_Y3_array = np.array([X3_f,Y3_f[:,0]])
    np.savetxt(store_path+'XY_l3_phi'+str(args.phi0_3)+'.dat',X3_Y3_array.T)
    rootslist3 = np.array([arr_x3,arr_list3[:,0],arr_list3[:,1]])
    np.savetxt(store_path+'roots3_phi'+str(args.phi0_3)+'.dat',rootslist3.T)
    M_data3 = np.array([X3_f,M_r3])
    np.savetxt(store_path+'M_l3_phi'+str(args.phi0_3)+'.dat',M_data3.T)
    Vc2_data3 = np.array([X3_units,Vc2_r3])
    np.savetxt(store_path+'Vc2_l3_phi'+str(args.phi0_3)+'.dat',Vc2_data3.T)
    ####sum
    Vc_tot_array = np.array([X0_units,Vc_tot])
    np.savetxt(store_path+'Vc_tot_l0123_phi0_'+str(args.phi0_0)+'phi1_'+str(args.phi0_1)+'phi2_'+str(args.phi0_2)+'phi3_'+str(args.phi0_3)+'.dat',Vc_tot_array.T)
time_f = time.time()
print("this process take:",time_f-time_i,"seconds")