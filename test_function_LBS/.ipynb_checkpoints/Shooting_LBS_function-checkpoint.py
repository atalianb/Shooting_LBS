import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
import argparse

my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-phi0',action='store',type=float,required=True)

args=my_parser.parse_args()
print(args.phi0)
###
###ODE system
###
def f(x,y,l=1.):
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
#the second correspond to the initial condition in epsilon (frequency)
###
def IC(u,k):
    return np.array([k,0.0,u[0],0.0,u[1]])
###
##Shooting method
###
def shooting(func,u,x0,x,xf,step,k,h=0.001):
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
##Values
####
u1_v01 = -3.7#5086524# 1st trial value of unknown init. cond.
u2_v01 = -2.2#1111819# 2nd trial value of unknown init. cond.
h = 0.001 # Step size
u_1v01 = np.array([u1_v01, u2_v01])
x0_1v01 = 0.0 # Start of integration
x_1v01 = 3.#first integration
xf_1v01 = 10. # End of integration
#u1_0 = -0.6#4957282 # 1st trial value of unknown init. cond.
#u2_0 = 1.5 # 2nd trial value of unknown init. cond.
step_0 = 0.5
###
###
X1_f,Y1_f,root1_f,arr_x1,arr_list1=shooting(f,u_1v01,x0_1v01,x_1v01,xf_1v01,step_0,k=args.phi0)
###
###
###Saving files
###
X1_Y1_array = np.array([X1_f,Y1_f[:,0]])
np.savetxt('X1_Y1.dat',X1_Y1_array.T)
rootslist1 = np.array([arr_x1,arr_list1[:,0],arr_list1[:,1]])
np.savetxt('roots1.dat',rootslist1.T)
####
###Ploting
####
plt.plot(X1_f,Y1_f[:,0],label='l=1')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right', prop={'size':10})
plt.savefig('X_Y'+str(args.phi0)+'.pdf')
####
###Integrated Mass function
####
def Mass_func(r,phi,l=1.):
    integral_temp = [0.,]
    phi_array = np.array(phi[:,0])
    for i in range(1,len(r)):
        temp_x = np.linspace(r[0],r[i],len(integral_temp)+1)
        temp_y = np.linspace(r[0],phi_array[i],len(integral_temp)+1)
        result = integrate.simps(temp_y**2.*temp_x**(2.*l+2.),temp_x)
        integral_temp.append(result)
    return np.array(integral_temp)
####
###Circular Speed function
####
def Vc2_cir(M,r):
    return M/r
###
##Mass
##
M_phi_0_01 = Mass_func(X1_f,Y1_f)
###
##
M_data01 = np.array([X1_f,M_phi_0_01])
np.savetxt('M_l1_phi_01.dat',M_data01.T)
####
##Plot
###
plt.plot(X1_f,M_phi_0_01,label=r'$\phi_{0}=0.1$')
plt.grid(True)
plt.legend(loc='upper right', prop={'size':10})
plt.xlabel("r")
plt.ylabel(r'$v_{c}^{2}$')
plt.title('Circular Speed 'r'$\ell = 1$')
plt.savefig('Mass_'+str(args.phi0)+'.pdf')
###
##Computing the circular speed
###
Vc2_phi_0_01 = Vc2_cir(M_phi_0_01,X1_f)
###
##Saving the information
###
Vc2_data01 = np.array([X1_f,Vc2_phi_0_01])
np.savetxt('Vc2_l1_phi_01.dat',Vc2_data01.T)
####
##Plot
###
plt.plot(X1_f,Vc2_phi_0_01,label=r'$\phi_{0}=0.1$')
plt.grid(True)
plt.legend(loc='upper right', prop={'size':10})
plt.xlabel("r")
plt.ylabel(r'$v_{c}^{2}$')
plt.title('Circular Speed 'r'$\ell = 1$')
plt.savefig('Vc2_'+str(args.phi0)+'.pdf')