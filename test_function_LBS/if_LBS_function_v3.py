#### to run:  python if_LBS_function.py -phi0 *phi_0 value* -l *l value*
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
import argparse
import time

time_i = time.time()

my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-phi0',action='store',type=float,required=True)
my_parser.add_argument('-l', action='store', type=int,required=True, choices=range(0,3))
#####
args=my_parser.parse_args()
print(args.phi0,args.l)
f_arg_l = float(args.l)
###
##Values
###
eps = 6.36e-5#value to be fixed
x0 = 0.01# Start of integration
xf = 10.# End of integration
h = 0.01# Step size of integration
step_0 = 0.5
###
###
###ODE system
###
def f(x,y,l=f_arg_l):
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
##
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
###
#Initial conditions array,
#the first correspond to the initial condition in phi
#the second correspond to the initial condition in omega (frequency)
###
def IC(u,k):
    return np.array([k,0.0,u[0],0.0,u[1]])
###
##Shooting method
###
def shooting(func,u,x0,x,xf,step,k,h):
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
####
###Integrated Mass function
####
def Mass_func(r,phi,l=f_arg_l):
    integral_temp = [0.,]
    phi_array = np.array(phi[:,0])
    for i in range(1,len(r)):
        temp_x = np.linspace(r[0],r[i],len(integral_temp)+1)
        temp_y = np.linspace(r[0],phi_array[i],len(integral_temp)+1)
        result = integrate.simps(temp_y**2.*temp_x**(2.*l+2.),temp_x)
        integral_temp.append(result)
    return np.array(integral_temp)
####
###Circular Speed function in (km/s)^2
####
def Vc2_cir(M,r):
    units = 3.90e-20*eps**2.
    return (units*M)/r
####
##Units for r in kpc
###
def r_units(r):
    return (6.36e-5*r)/eps
###
###
if f_arg_l==0.:
    u1_0 = 0.63173215# 1st trial value of unknown init. cond.
    u2_0 = 1.28125868# 2nd trial value of unknown init. cond.
    u_0 = np.array([u1_0, u2_0])
    x0_0 = 2.#first integration
    X0_f,Y0_f,root0_f,arr_x0,arr_list0=shooting(f,u_0,x0,x0_0,xf,step_0,k=args.phi0,h=0.01)
    ###
    ###Saving files
    ###
    X0_Y0_array = np.array([X0_f,Y0_f[:,0]])
    np.savetxt('XY_l'+str(args.l)+'phi'+str(args.phi0)+'.dat',X0_Y0_array.T)
    rootslist0 = np.array([arr_x0,arr_list0[:,0],arr_list0[:,1]])
    np.savetxt('roots_l'+str(args.l)+'phi0_'+str(args.phi0)+'.dat',rootslist0.T)
    ###
    ##Mass
    ##
    M_r0 = Mass_func(X0_f,Y0_f)
    ###
    ##
    M_data0 = np.array([X0_f,M_r0])
    np.savetxt('M_l'+str(args.l)+'phi0_'+str(args.phi0)+'.dat',M_data0.T)
    ###
    ##Computing the circular speed
    ###
    Vc2_r0 = Vc2_cir(M_r0,X0_f)
    ###
    ##Computing r in kpc
    ###
    X0_units = r_units(X0_f)
    ###
    ##Saving the information
    ###
    Vc2_data0 = np.array([X0_units,Vc2_r0])
    np.savetxt('Vc2_l'+str(args.l)+'phi0_'+str(args.phi0)+'.dat',Vc2_data0.T)
    ####
    ####
    ##Plot of the circular speed
    ###
    plt.figure()
    plt.plot(X0_units,Vc2_r0,label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel("r[kpc]")
    plt.ylabel(r'$v_{c}^{2}[(km/s)^{2}]$')
    plt.title('Circular Speed 'r'$\ell =$'+str(args.l))
    plt.savefig('Vc2_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
    ###Plot of the integration
    ####
    plt.figure()
    plt.plot(X0_f,Y0_f[:,0],label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Integration result 'r'$\ell =$'+str(args.l))
    plt.legend(loc='upper right', prop={'size':10})
    plt.savefig('X_Y_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
    ##Plot of the mass function
    ###
    plt.figure()
    plt.plot(X0_f,M_r0,label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel("r")
    plt.ylabel(r'$M(r)$')
    plt.title('Mass function 'r'$\ell = $'+str(args.l))
    plt.savefig('Mass_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    
if f_arg_l==1.:
    u1_1 = -3.75086524# 1st trial value of unknown init. cond.
    u2_1 = -2.21111819# 2nd trial value of unknown init. cond.
    u_1 = np.array([u1_1, u2_1])
    x_1 = 3.#first integration
    X1_f,Y1_f,root1_f,arr_x1,arr_list1=shooting(f,u_1,x0,x_1,xf,step_0,k=args.phi0,h=0.01)
    ###
    ###Saving files
    ###
    X1_Y1_array = np.array([X1_f,Y1_f[:,0]])
    np.savetxt('XY_l'+str(args.l)+'phi'+str(args.phi0)+'.dat',X1_Y1_array.T)
    rootslist1 = np.array([arr_x1,arr_list1[:,0],arr_list1[:,1]])
    np.savetxt('roots_l'+str(args.l)+'phi0_'+str(args.phi0)+'.dat',rootslist1.T)
    ###
    ##Mass
    ##
    M_r1 = Mass_func(X1_f,Y1_f)
    ###
    ##
    M_data1 = np.array([X1_f,M_r1])
    np.savetxt('M_l'+str(args.l)+'phi0_'+str(args.phi0)+'.dat',M_data1.T)
    ###
    ##Computing the circular speed
    ###
    Vc2_r1 = Vc2_cir(M_r1,X1_f)
    ###
    ##Computing r in kpc
    ###
    X1_units = r_units(X1_f)
    ###
    ##Saving the information
    ###
    Vc2_data1 = np.array([X1_units,Vc2_r1])
    np.savetxt('Vc2_l'+str(args.l)+'phi0_'+str(args.phi0)+'.dat',Vc2_data1.T)
    ####
    ####
    ####
    ##Plot of the circular speed
    ###
    plt.figure()
    plt.plot(X1_units,Vc2_r1,label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel("r[kpc]")
    plt.ylabel(r'$v_{c}^{2}[(km/s)^{2}]$')
    plt.title('Circular Speed 'r'$\ell =$'+str(args.l))
    plt.savefig('Vc2_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
    ###Plot of the integration
    ####
    plt.figure()
    plt.plot(X1_f,Y1_f[:,0],label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Integration result 'r'$\ell = $'+str(args.l))
    plt.legend(loc='upper right', prop={'size':10})
    plt.savefig('X_Y_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
    ##Plot of the mass function
    ###
    plt.figure()
    plt.plot(X1_f,M_r1,label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel("r")
    plt.ylabel(r'$M(r)$')
    plt.title('Mass function 'r'$\ell = $'+str(args.l))
    plt.savefig('Mass_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
if f_arg_l==2.:
    u1_2 = -3.75086524# 1st trial value of unknown init. cond.
    u2_2 = -2.21111819# 2nd trial value of unknown init. cond.
    u_2 = np.array([u1_2, u2_2])
    x_2 = 3.#first integration
    X2_f,Y2_f,root2_f,arr_x2,arr_list2=shooting(f,u_2,x0,x_2,xf,step_0,k=args.phi0,h=0.01)
    ###
    ###Saving files
    ###
    X2_Y2_array = np.array([X2_f,Y2_f[:,0]])
    np.savetxt('XY_l'+str(args.l)+'phi'+str(args.phi0)+'.dat',X2_Y2_array.T)
    rootslist2 = np.array([arr_x2,arr_list2[:,0],arr_list2[:,1]])
    np.savetxt('roots'+str(args.l)+'phi'+str(args.phi0)+'.dat',rootslist2.T)
    ###
    ##Mass
    ##
    M_r2 = Mass_func(X2_f,Y2_f)
    ###
    ##
    M_data2 = np.array([X2_f,M_r2])
    np.savetxt('M_l'+str(args.l)+'phi'+str(args.phi0)+'.dat',M_data2.T)
    ###
    ##Computing the circular speed
    ###
    Vc2_r2 = Vc2_cir(M_r2,X2_f)
    ###
    ##Computing r in kpc
    ###
    X2_units = r_units(X2_f)
    ###
    ##Saving the information
    ###
    Vc2_data2 = np.array([X2_units,Vc2_r2])
    np.savetxt('Vc2_l'+str(args.l)+'phi'+str(args.phi0)+'.dat',Vc2_data2.T)
    ####
     ####
    ##Plot of the circular speed
    ###
    plt.figure()
    plt.plot(X2_units,Vc2_r2,label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel("r[kpc]")
    plt.ylabel(r'$v_{c}^{2}[(km/s)^{2}]$')
    plt.title('Circular Speed 'r'$\ell = $'+str(args.l))
    plt.savefig('Vc2_+'+str(args.l)+'phi'+str(args.phi0)+'.pdf')
    ####
    ###Plot of the integration
    ####
    plt.figure()
    plt.plot(X2_f,Y2_f[:,0],label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Integration result 'r'$\ell = $'+str(args.l))
    plt.legend(loc='upper right', prop={'size':10})
    plt.savefig('X_Y_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
    ##Plot of the mass function
    ###
    plt.figure()
    plt.plot(X2_f,M_r2,label=r'$\phi_{0}=$'+str(args.phi0))
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel("r")
    plt.ylabel(r'$M(r)$')
    plt.title('Mass function 'r'$\ell = $'+str(args.l))
    plt.savefig('Mass_l'+str(args.l)+'phi0_'+str(args.phi0)+'.pdf')
    ####
    
time_f = time.time()
print("this process take:",time_f-time_i,"seconds")