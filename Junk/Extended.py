#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:38:24 2019

@author: mhyip
"""

import numpy as np
from mpi4py import MPI

#%%%
comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
#%%
import sympy
#sympy.init_printing()

z = sympy.symbols('z')
c= sympy.symbols('c')
K0 = sympy.symbols('K0')
K1 = sympy.symbols('K1')
g = sympy.symbols('g')
Zb= sympy.symbols('Zb')
w=0

sym_Diffu =  (K0 + K1 * z * sympy.exp(-g * z))*(sympy.tanh(c*(z-Zb)))
sym_dKdz = sympy.diff(sym_Diffu, z, 1)
sym_Beta = sympy.sqrt(2 * sym_Diffu)
sym_dBdz = sympy.diff(sym_Beta, z, 1)
sym_ddBdzz = sympy.diff(sym_Beta, z, 2)
sym_Alpha = w + sym_dKdz
sym_dAdz = sympy.diff(sym_Alpha, z, 1)
sym_ddAdzz = sympy.diff(sym_Alpha, z, 2)
sym_dABdz = sympy.diff(sym_Alpha * sym_Beta, z, 1)

Diffu  =  sympy.utilities.lambdify(z,        sym_Diffu,np)
dKdz   =  sympy.utilities.lambdify(z,         sym_dKdz,np)
Beta   =  sympy.utilities.lambdify(z,         sym_Beta,np)
dBdz   =  sympy.utilities.lambdify(z,         sym_dBdz,np)
ddBdzz=  sympy.utilities.lambdify(z,        sym_ddBdzz,np)
Alpha =  sympy.utilities.lambdify(z,      sym_Alpha,np)
dAdz  =  sympy.utilities.lambdify(z,      sym_dAdz,np)
ddAdzz=  sympy.utilities.lambdify(z,      sym_ddAdzz,np)
dABdz =  sympy.utilities.lambdify(z, sym_Alpha*sym_Beta,np)


#%%
K0 = 1e-3# m * * 2 / s
K1 = 6e-3# m / s
g = 0.5
Zb=-0.05
c=100
w=0


Diffu  =  lambda z : (K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z))

dKdz   =  lambda z : c*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1)\
                     + (-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*np.tanh(c*(-Zb + z))
                     
Beta   =  lambda z : np.sqrt(2)*np.sqrt((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z)))

dBdz   =  lambda z : np.sqrt(2)*np.sqrt((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z)))\
                    *(c*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1)/2\
                    + (-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*np.tanh(c*(-Zb + z))/2)\
                      /((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z)))
                      
ddBdzz =  lambda z : np.sqrt(2)*np.sqrt(-(K0 + K1*z*np.exp(-g*z))*np.tanh(c*(Zb - z)))\
                     *(-K1*c*(g*z - 1)*(np.tanh(c*(Zb - z))**2 - 1)*np.exp(-g*z)\
                     + K1*g*(g*z - 2)*np.exp(-g*z)*np.tanh(c*(Zb - z))/2\
                     - K1*(g*z - 1)*(K1*(g*z - 1)*np.exp(-g*z)*np.tanh(c*(Zb - z))\
                     - c*(K0 + K1*z*np.exp(-g*z))*(np.tanh(c*(Zb - z))**2 - 1))\
                    *np.exp(-g*z)/(2*(K0 + K1*z*np.exp(-g*z))) + c**2*(K0 + K1*z*np.exp(-g*z))\
                    *(np.tanh(c*(Zb - z))**2 - 1)*np.tanh(c*(Zb - z)) + c*(K1*(g*z - 1)\
                      *np.exp(-g*z)*np.tanh(c*(Zb - z)) - c*(K0 + K1*z*np.exp(-g*z))\
                      *(np.tanh(c*(Zb - z))**2 - 1))*(np.tanh(c*(Zb - z))**2 - 1)\
                      /(2*np.tanh(c*(Zb - z))) + (K1*(g*z - 1)*np.exp(-g*z)*np.tanh(c*(Zb - z))\
                        - c*(K0 + K1*z*np.exp(-g*z))\
                        *(np.tanh(c*(Zb - z))**2 - 1))**2\
                        /(4*(K0 + K1*z*np.exp(-g*z))*np.tanh(c*(Zb - z))))\
                        /((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(Zb - z)))
    
    
Alpha  =  lambda z : c*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1) \
                        + (-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*np.tanh(c*(-Zb + z))
dAdz   =  lambda z : -2*c**2*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1)*np.tanh(c*(-Zb + z)) + 2*c*(-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1) + (K1*g**2*z*np.exp(-g*z) - 2*K1*g*np.exp(-g*z))*np.tanh(c*(-Zb + z))
ddAdzz =  lambda z : 6*K1*c**2*(g*z - 1)*(np.tanh(c*(Zb - z))**2 - 1)*np.exp(-g*z)*np.tanh(c*(Zb - z)) - 3*K1*c*g*(g*z - 2)*(np.tanh(c*(Zb - z))**2 - 1)*np.exp(-g*z) + K1*g**2*(g*z - 3)*np.exp(-g*z)*np.tanh(c*(Zb - z)) - 2*c**3*(K0 + K1*z*np.exp(-g*z))*(np.tanh(c*(Zb - z))**2 - 1)**2 - 4*c**3*(K0 + K1*z*np.exp(-g*z))*(np.tanh(c*(Zb - z))**2 - 1)*np.tanh(c*(Zb - z))**2
dABdz  =  lambda z : np.sqrt(2)*np.sqrt((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z)))*(-2*c**2*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1)*np.tanh(c*(-Zb + z)) + 2*c*(-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1) + (K1*g**2*z*np.exp(-g*z) - 2*K1*g*np.exp(-g*z))*np.tanh(c*(-Zb + z))) + np.sqrt(2)*np.sqrt((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z)))*(c*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1)/2 + (-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*np.tanh(c*(-Zb + z))/2)*(c*(K0 + K1*z*np.exp(-g*z))*(-np.tanh(c*(-Zb + z))**2 + 1) + (-K1*g*z*np.exp(-g*z) + K1*np.exp(-g*z))*np.tanh(c*(-Zb + z)))/((K0 + K1*z*np.exp(-g*z))*np.tanh(c*(-Zb + z)))
#%%
#######
#Euler#
#######
def step_e(z,H,dt,N_sample):
    
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    a=dKdz(z)
    b=np.sqrt(2*Diffu(z))
    
    temp=z+a*dt+b*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_e_Test1(z,H,dt,N_sample):
    
    
    a=dKdz(z)
    
    temp=z+a*dt
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_e_Test2(z,H,dt,N_sample):
    
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    b=np.sqrt(2*Diffu(z))
    
    temp=z+b*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_e_const(z,H,dt,N_sample):
    
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    K0=3e-3
    b=np.sqrt(2*K0)
    
    temp=z+b*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

########
#Visser#
########
def step_v(z,H,dt,N_sample):
    
    #dW=np.random.uniform(-1,1,N_sample)
    #r=1/3
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    r=dt
    
    a=dKdz(z)
    G=Diffu(z+a*dt/2)
    
    temp= z + a*dt + np.sqrt(2/r*dt*G)*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_v_const(z,H,dt,N_sample):
    K0=3e-3
    
    #dW=np.random.uniform(-1,1,N_sample)
    #r=1/3
    
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    r=dt
    
    temp= z + np.sqrt(2/r*dt*K0)*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

##############
#Milstein 1nd#
##############
def step_m(z,H,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    #de=np.random.normal(dt,np.sqrt(dt),N_sample)
    #temp= z + (1/2)*dKdz(z)*(de+dt) + Beta(z)*dW
    
    temp= z + (1/2)*dKdz(z)*(dW*dW+dt) + Beta(z)*dW
    temp=np.where(temp<Zb, Zb+(Zb-temp) ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

##############
#Milstein 2nd#
##############
def step_m2(z,H,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    k=Diffu(z)
    dkdz=dKdz(z)
    ddkdz=dAdz(z)
    dddkdz=ddAdzz(z)
    sqrt2k=np.sqrt(2*k)
    
    a=dkdz
    da=ddkdz
    dda=dddkdz
    b= sqrt2k 
    db=dkdz/b
    ddb=ddkdz/b - ((dkdz)**2)/b**3
    ab=da*b+a*db
    
    temp= z + a*dt+b*dW+1/2*b*db*(dW*dW-dt)+1/2*(ab+1/2*ddb*b**2)*dW*dt+\
            1/2*(a*da+1/2*dda*b**2)*dt**2
    
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_m2_const(z,H,dt,N_sample):
    
    K0=3e-3
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    temp= z +np.sqrt(2*K0)*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

#%%
def oneStep(upperBound, lowerBound, TimeAdaptive, scheme, rFactor ,H, dt):
    
    maskBody=((upperBound<TimeAdaptive)&(lowerBound>TimeAdaptive))

    zBody=TimeAdaptive[maskBody]
    zBody=scheme(zBody, H, dt, zBody.size)

    zEdge=TimeAdaptive[~maskBody]
    for j in range(rFactor):
        zEdge=scheme(zEdge,H,dt/rFactor,zEdge.size)

    TimeAdaptive[maskBody]=zBody
    TimeAdaptive[~maskBody]=zEdge
    
    return None

def parallelBodyEdge(Tmax, dt, H, prosent, rFactor, Testdepth, Np, Nbins):
    np.random.seed()
    upperBound =H*prosent
    lowerBound =H-H*prosent
    Ntime   =int(Tmax/dt)  

    hist_E_timeAdap      =np.zeros((Nbins-1,),'i')
    hist_V_timeAdap      =np.zeros((Nbins-1,),'i')
    hist_M_timeAdap      =np.zeros((Nbins-1,),'i')
    hist_M2_timeAdap     =np.zeros((Nbins-1,),'i')
    hist_EConst_timeAdap =np.zeros((Nbins-1,),'i')
    hist_VConst_timeAdap =np.zeros((Nbins-1,),'i')
    hist_MConst_timeAdap =np.zeros((Nbins-1,),'i')

    zEulTimeAdaptive    =np.random.uniform(0,H,int(Np))
    zVisTimeAdaptive    =np.random.uniform(0,H,int(Np))
    zMilTimeAdaptive    =np.random.uniform(0,H,int(Np))
    zMil2TimeAdaptive   =np.random.uniform(0,H,int(Np))
    zEulConTimeAdaptive =np.random.uniform(0,H,int(Np))
    zVisConTimeAdaptive =np.random.uniform(0,H,int(Np))
    zMilConTimeAdaptive =np.random.uniform(0,H,int(Np))

    for i in range(Ntime):

        oneStep(upperBound, lowerBound, zEulTimeAdaptive,    step_e,        rFactor ,H, dt)
        oneStep(upperBound, lowerBound, zVisTimeAdaptive,    step_v,        rFactor ,H, dt)
        oneStep(upperBound, lowerBound, zMilTimeAdaptive,    step_m,        rFactor ,H, dt)
        oneStep(upperBound, lowerBound, zMil2TimeAdaptive,   step_m2,       rFactor ,H, dt)
        oneStep(upperBound, lowerBound, zEulConTimeAdaptive, step_e_const,  rFactor ,H, dt)
        oneStep(upperBound, lowerBound, zVisConTimeAdaptive, step_v_const,  rFactor ,H, dt)
        oneStep(upperBound, lowerBound, zMilConTimeAdaptive, step_m2_const, rFactor ,H, dt)

        #Adding the histogram
        ###
        temp0, _ = np.histogram(zEulTimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_E_timeAdap=hist_E_timeAdap + temp0

        temp1, _ = np.histogram(zVisTimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_V_timeAdap=hist_V_timeAdap + temp1

        temp2, _ = np.histogram(zMilTimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_M_timeAdap=hist_M_timeAdap + temp2

        temp3, _ = np.histogram(zMil2TimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_M2_timeAdap=hist_M2_timeAdap + temp3

        ### Constant potensial
        temp4, _ = np.histogram(zEulConTimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_EConst_timeAdap=hist_EConst_timeAdap+temp4

        temp5, _ = np.histogram(zVisConTimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_VConst_timeAdap=hist_VConst_timeAdap+temp5

        temp6, _ = np.histogram(zMilConTimeAdaptive, bins = np.linspace(0, Testdepth, Nbins))
        hist_MConst_timeAdap=hist_MConst_timeAdap+temp6

        if (i % int(Ntime/100) ==0):
            print("\r %6.2f"% (i*100/Ntime+1),"%", end="\r",flush=True)
            
#    queneEu.put(hist_E_timeAdap)
#    queneEuConst.put(hist_EConst_timeAdap)
#    queneV.put(hist_V_timeAdap)
#    queneVConst.put(hist_VConst_timeAdap)
#    queneM.put(hist_M_timeAdap)
#    queneM2.put(hist_M2_timeAdap)
#    queneM2Const.put(hist_MConst_timeAdap)
            
    return hist_M_timeAdap
        

#%%
Np        = 400000
Nbins     = 600
Tmax      = 2*3600              #Maximum time
dt        = 0.001               #Delta time
H         = 10
prosent   = 0.1
rFactor   = 1
Testdepth = 10

sub_Np= Np / size

residue = Np % size

if(rank < residue):
    sub_Np= sub_Np+1

hist_M_timeAdap = parallelBodyEdge(Tmax, dt, H, prosent, rFactor, Testdepth, sub_Np, Nbins)

comm.barrier()

temp_hist_m=hist_M_timeAdap.copy()
comm.Reduce(temp_hist_m,hist_M_timeAdap,op=MPI.SUM,root=0)

if (rank==0):
    np.save('hist.npy', hist_M_timeAdap)
