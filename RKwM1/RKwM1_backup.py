#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:18:54 2019

@author: mhyip
"""

#from mpi4py import MPI
import numpy as np
import warnings
warnings.filterwarnings("error")
import sys
#matplotlib inline
from matplotlib import pyplot as plt
# Get nicer looking plots than default
plt.style.use('bmh')
# Initialize MPI (this is equivalent to MPI_INIT in C or Fortran)
#comm = MPI.COMM_WORLD
# Get the rank of this process (this is euivalent to MPI_COMM_RANK in C or Fortran)
#rank = comm.Get_rank()

#np.random.seed(225)

def drift(z):
    return z/10
    # return -0.003*z*np.exp(-0.5*z) + 0.006*np.exp(-0.5*z)

def diffu(z):
    return z/20
    #return np.sqrt(2*3e-3)
    # return np.sqrt(0.012*z*np.exp(-0.5*z) + 0.002)


# %%
# Here we are creating the butcher table for our Rossler SRK method.
# See page 37 (Rackauckas)
ZerosMatix = np.zeros((4, 4), dtype="float64")

c0 = np.array([0, 3/4, 0, 0])
a0 = ZerosMatix.copy()
a0[1, 0] = 3/4
b0 = ZerosMatix.copy()
b0[1, 0] = 3/2

c1 = np.array([0, 1/4, 1, 1/4])
a1 = ZerosMatix.copy()
a1[1, 0] = 1/4
a1[2, 0] = 1.0
a1[3, 2] = 1/4
b1 = ZerosMatix.copy()
b1[1, 0] = 1/2
b1[2, 0] = -1.0
b1[3, 0] = -5.0
b1[3, 1] = 3.0
b1[3, 2] = 1/2

alp = np.array([1/3, 2/3, 0, 0])
beta = ZerosMatix.copy()
beta[0, :] = [-1.0, 4/3, 2/3, 0.0]
beta[1, :] = [-1.0, 4/3, -1/3, 0.0]
beta[2, :] = [2.0, -4/3, -2/3, 0.0]
beta[3, :] = [-2.0, 5/3, -2/3, 1.0]

alpTil = np.array([1/2, 1/2, 0, 0])
betaTil = beta.copy()
betaTil[2, :] = 0
betaTil[3, :] = 0

del ZerosMatix

# %%
# Here we difine the Rossler SRK method.


def SRK_1_5(z, dt, dW, dV):

    # The first Brownian walk (dW)
    #dW = np.random.normal(0,np.sqrt(dt))
    # The second Brownian walk (zeta) for calculate I10. See page 24 (Rackauckas)
    #dV = np.random.normal(0,np.sqrt(dt))

    # Wiktorsson iterated stochastic integral approximations (Ito sense).
    I1 = dW
    I11 = 1/2*(I1*I1 - dt)
    I10 = 1/2*dt*(I1 + (1/np.sqrt(3))*dV)
    I111 = 1/6*(I1*I1*I1 - 3*dt*I1)

    # Calculate predictor.
    H0 = np.zeros((4,), dtype="float64")
    H1 = np.zeros((4,), dtype="float64")
    for i in range(4):
        H0[i] = z + np.sum(a0[i, :]*drift(H0))*dt + np.sum(b0[i, :]*diffu(H1))*(I10/dt)
            
        H1[i] = z + np.sum(a1[i, :]*drift(H0))*dt + np.sum(b1[i, :]*diffu(H1))*np.sqrt(dt)

    # Because the 1.0 is embadded. It will be wise to first calculate the error, then calculate 1.5 SRK.
    # Some of term will be use to calculate the error.
    # Equation 9 in (Rackauckas)
    E1 = drift(H0[0])
    E2 = drift(H0[1])
    E3 = np.sum((beta[2, :]*I10/dt + beta[3, :]*I111/dt)*diffu(H1))
#    temp = (beta[2, :]*I10/dt + beta[3, :]*I111/dt)
#    if (E3 == 0.0):
#        print("Something wrong")
#        print("E3 :", E3)
#        print("dt: ", dt)
#        print("dW: ", dW)
#        print("dV: ", dV)
#        print("I10: ", I10)
#        print("I111: ", I111)        
#        print("I10/dt: ", I10/dt)
#        print("I111/dt: ", I111/dt)
#        #print("beta[2,:] :", beta[2, :])
#        #print("beta[3,:] :", beta[3, :])
#        print("H1: ", H1)
#        print("diffu(H1)", diffu(H1))
#        print("temp: ", temp)
#        print("SUM: ", np.sum(temp))
#        print("loglic: ", np.sum(temp)==0.0)
        
    Error = -(dt/6)*E1 + (dt/6)*E2 + E3

    # Calculate nest step with dt with Rossler SRK method. See equation 2 in (Rackauckas).
    # Here we utilise some of the valuse that have been alreday calculated.
    DRIFT = (alp[0]*E1 + alp[1]*E2 + alp[2]*drift(H0[2]) + alp[3]*drift(H0[3]))*dt
             
    DIFFU = np.sum((beta[0, :]*I1 + beta[1, :]*I11/(np.sqrt(dt)))*diffu(H1)) + E3
    zNew = z + DRIFT + DIFFU

    # reflecting boundary condition.
    H = 10.0
    zRef = 0.0
    zRef = zNew
    #zRef = np.where(zRef < 0, -zRef, zRef)
    #zRef = np.where(zRef > H, 2*H-zRef, zRef)

    return zRef, zNew, Error

# %%


def simulation():
    # See page 11 (Rackauckas)
    # Setting some valuse
    epsilon_abs = 1e-5
    epsilon_rel = 1e-5
    epsilon = np.array([epsilon_abs, epsilon_rel])
    T_end = 2
    dt_max = 0.1
    gamma = 5
    MiniOrder = 1
    t = 0
    W = 0
    V = 0
    #z = np.random.uniform(0, 10)
    z = 0.5
    Stack = []

    # Initialise
    dt = dt_max
    dW = np.random.normal(0, np.sqrt(dt))
    dV = np.random.normal(0, np.sqrt(dt))
    z_regis = np.array([z])
    T_regis = np.array([0.0])
    W_regis = np.array([0.0])
    dW_regis = np.array([dW])
    
    while (t < (T_end - 1e-12)):

        # print(dt)
        Temp_zRef, Temp_z, Error = SRK_1_5(z, dt, dW, dV)

        # See page 8 (Rackauckas)
        sc = epsilon[0] + Temp_zRef*epsilon[1]
        e = np.abs(Error/sc)
        #print(e)
#        q = np.power(1/(gamma*e), 1/(MiniOrder+1))
        try:
            q = np.power(1/(gamma*e), 1/(MiniOrder+1))
        except:
            print("t:", t)
            print("Error: ", Error)
            print("sc: ", sc)
            print("e: ", e)
            sys.exit()

        if (q < 1):  # Reject the step
            
            dW_tilde = np.random.normal(q*dW, np.sqrt((1-q)*q*dt))
            dV_tilde = np.random.normal(q*dV, np.sqrt((1-q)*q*dt))

            dW_bar = dW - dW_tilde
            dV_bar = dV - dV_tilde

            # Put to memory
            Stack.insert(0, np.array([(1-q)*dt, dW_bar, dV_bar]))

            # Update some values
            dt = q*dt
            dW = dW_tilde
            dV = dV_tilde

        else:  # Accept the step
            # Update
            t = t + dt
            W = W + dW
            V = V + dV
            z = Temp_zRef
            T_regis = np.append(T_regis, t)
            z_regis = np.append(z_regis, z)
            W_regis = np.append(W_regis, W)
            
            if(not Stack): 
#                hist, _ = np.histogram(z_regis, bins=np.linspace(0, 10, 100))
#                T_regis = np.append(T_regis, t)
#                z_regis = np.append(z_regis, z)
#                W_regis = np.append(W_regis, W)

                # Update
                c = min(dt_max, q*dt)
                #c = dt_max
                rest = T_end - t
                dt = min(c, rest)
                
                dW = np.random.normal(0, np.sqrt(dt))
                dV = np.random.normal(0, np.sqrt(dt))
            else:
                # Update
                L = Stack.pop(0)
                dt = L[0]
                dW = L[1]
                dV = L[2]
    
    
    return hist, z_regis, T_regis, W_regis, dW_regis

# %%
Np = 1
hist = 0
for i in range(Np):
    temp, z, t, W, dW = simulation()
    hist = hist + temp

# Use Reduce to calculate sum of concentrations on rank 0
# First create an additional array on rank 0
#if rank == 0:
#    hist_global = np.zeros_like(hist)
# Make a dummy on all other ranks
#else:
#    hist_global = None
# First argument is sendbuf, second is recvbuf
#comm.Reduce(hist, hist_global, op=MPI.SUM, root=0)

# Store the results on rank 0 only
#if rank == 0:
#np.save('hist.npy', hist_global)

#print(hist)
#z = z[1:]-z[:-1]
#print("z: \n" , z)
#t = t[1:]-t[:-1]
#print("t: \n", t)
#%%
def testfunction(t, W):
    beta = 1/20
    alpha = 1/10
    c = (beta-(alpha**2)/2)
    print("c :" ,c)
    z = 0.5*np.exp(c*t + alpha*W)
    return z

def Euler(t, W):
    nt = t.size
    z = np.zeros(nt)
    z[0] = 0.5
    for i in range(nt-1):
        z[i+1] = z[i] + drift(z[i])*(t[i+1] - t[i]) + diffu(z[i])*(W[i+1]-W[i])
    return z

#%%
plt.figure(1)
#plt.plot(t, ".")
plt.plot(t, W, "-*")

plt.figure(2)
trueZ = testfunction(t, W)
EulerZ = Euler(t, W)
plt.plot(t, trueZ, label = "True")
plt.plot(t, EulerZ, ".", label = "Euler")
plt.plot(t, z, "*")
plt.legend()
#
#plt.figure(3)
#plt.plot(t[1:]-t[:-1], "*")


plt.show()