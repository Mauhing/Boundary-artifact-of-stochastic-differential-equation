#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:21:18 2019

@author: mhyip
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('bmh')
#import datetime
#import threading
#import multiprocessing as mp
#from matplotlib import pyplot as plt
#from time import time
#from scipy.stats import norm

import warnings
warnings.filterwarnings("error")
#%%
import sympy
#sympy.init_printing()
z = sympy.symbols('z')
K0 = 1e-3# m * * 2 / s
K1 = 6e-3# m / s
Aalpha = 0.5
c=20
w=0

sym_Diffu =  K0 + K1 * z * sympy.exp(-Aalpha * z)# + K1/c*(1-sympy.tanh(c*z))
#sym_Diffu =  2e-3*sympy.cos(4/10*2*np.pi*z) + 2*2e-3
sym_dKdz = sympy.diff(sym_Diffu, z, 1)
sym_Beta = sympy.sqrt(2 * sym_Diffu)
sym_dBdz = sympy.diff(sym_Beta, z, 1)
sym_ddBdzz = sympy.diff(sym_Beta, z, 2)
sym_Alpha = w + sym_dKdz
sym_dAdz = sympy.diff(sym_Alpha, z, 1)
sym_ddAdzz = sympy.diff(sym_Alpha, z, 2)
sym_dABdz = sympy.diff(sym_Alpha * sym_Beta, z, 1)

K  =  sympy.utilities.lambdify(z,          sym_Diffu,np)
dKdz   =  sympy.utilities.lambdify(z,          sym_dKdz,np)
diffu   =  sympy.utilities.lambdify(z,          sym_Beta,np)
dBdz   =  sympy.utilities.lambdify(z,          sym_dBdz,np)
ddBdzz=  sympy.utilities.lambdify(z,          sym_ddBdzz,np)
drift =  sympy.utilities.lambdify(z,      sym_Alpha,np)
dAdz  =  sympy.utilities.lambdify(z,      sym_dAdz,np)
ddAdzz=  sympy.utilities.lambdify(z,      sym_ddAdzz,np)
dABdz =  sympy.utilities.lambdify(z, sym_Alpha*sym_Beta,np)

print(sym_Alpha)
#%%
s = np.linspace(0,10, 1000)
plt.plot(s, drift(s))

#%%
#Here we are creating the butcher table for our Rossler SRK method.
#See page 37 (Rackauckas)
ZerosMatix = np.zeros((4,4), dtype="float64")

c0 = np.array([0, 3/4, 0, 0])
a0 = ZerosMatix.copy(); a0[1,0] = 3/4
b0 = ZerosMatix.copy(); b0[1,0] = 3/2

c1 = np.array([0, 1/4, 1, 1/4])
a1 = ZerosMatix.copy(); a1[1,0] = 1/4; a1[2,0] = 1; a1[3,2] = 1/4; 
b1 = ZerosMatix.copy(); b1[1,0] = 1/2; b1[2,0] = -1; b1[3,0] = -5; b1[3,1] = 3; b1[3,2] = 1/2;

alp = np.array([1/3, 2/3, 0, 0])
beta= ZerosMatix.copy(); 
beta[0, :] = [-1, 4/3, 2/3, 0]
beta[1, :] = [-1, 4/3, -1/3, 0]
beta[2, :] = [2, -4/3, -2/3, 0]
beta[3, :] = [-2, 5/3, -2/3, 1]

alpTil = np.array([1/2, 1/2, 0, 0])
betaTil= beta.copy(); 
betaTil[2, :] = 0
betaTil[3, :] = 0

del ZerosMatix 
#%%
#Here we difine the Rossler SRK method.
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
    
    #Calculate predictor.
    H0 = np.zeros((4,), dtype="float64")
    H1 = np.zeros((4,), dtype="float64")
    for i in range(4):
        H0[i] = z + np.sum(a0[i,:]*drift(H0))*dt + np.sum(b0[i,:]*diffu(H1))*I10/dt
        H1[i] = z + np.sum(a1[i,:]*drift(H0))*dt + np.sum(b1[i,:]*diffu(H1))*np.sqrt(dt)

    #Because the 1.0 is embadded. It will be wise to first calculate the error, then calculate 1.5 SRK. 
    #Some of term will be use to calculate the error. 
    #Equation 9 in (Rackauckas)
    E1 = drift(H0[0])
    E2 = drift(H0[1])
    E3 = np.sum((beta[2,:]*I10/dt + beta[3,:]*I111/dt)*diffu(H1))
    Error = -dt/6*E1 + dt/6*E2 +E3
    
    #Calculate nest step with dt with Rossler SRK method. See equation 2 in (Rackauckas).
    #Here we utilise some of the valuse that have been alreday calculated. 
    DRIFT = (alp[0]*E1 + alp[1]*E2 + alp[2]*drift(H0[2]) + alp[3]*drift(H0[3]))*dt
    DIFFU = np.sum((beta[0,:]*I1 + beta[1,:]*I11/(np.sqrt(dt)))*diffu(H1)) + E3
    zNew = z + DRIFT + DIFFU
    
    #reflecting boundary condition.
    H = 10
    zRef = zNew.copy()
    zRef = np.where(zRef<0, -zRef ,zRef)
    zRef = np.where(zRef>H, 2*H-zRef,zRef)

    return zRef, zNew, Error
#%%
# See page 11 (Rackauckas)
# Setting some valuse
epsilon_abs = 1e-5; epsilon_rel = 1e-5
epsilon = np.array([epsilon_abs, epsilon_rel]) 
dt_max = 1; T_end = 2*3600
gamma = 2; MiniOrder = 1
t = 0; W = 0; V = 0; z = 1 #np.random.uniform(0, 10)
Stack = []

# Initialise
dt = 10.0
dW = np.random.normal(0,np.sqrt(dt))
dV = np.random.normal(0,np.sqrt(dt))

while (t<T_end):
    
    Temp_zRef, Temp_z, Error = SRK_1_5(z, dt, dW, dV)
    
    #See page 8 (Rackauckas)
    sc = epsilon[0] + z*epsilon[1]
    e = np.sqrt(np.abs(Error)/sc)
    q = np.power(1/(gamma*e),1/(MiniOrder+1))
    
    if (q < 1): # Reject the step
        dW_tilde = np.random.normal(q*dW, np.sqrt((1-q)*q*dt))
        dV_tilde = np.random.normal(q*dV, np.sqrt((1-q)*q*dt))
        
        dW_bar = dW - dW_tilde
        dV_bar = dV - dV_tilde
        
        #Put to memory
        Stack.insert(0, np.array([(1-q)*dt, dW_bar, dV_bar]))
        
        #Update some values
        dt = q*dt
        dW = dW_tilde
        dV = dV_tilde
    else: # Accept the step
        #Update
        t = t + dt; W = W + dW; V = V + dV; z = Temp_zRef
        if(not Stack):
            #Update
            c = min(dt_max, q*dt)
            dt = min(c, T_end-t)
        else:
            #Update
            L = Stack.pop(0)
            dt=L[0]
            dW=L[1]
            dV=L[2]
