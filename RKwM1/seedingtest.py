#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:37:21 2019

@author: mhyip
"""

import numpy as np
np.random.seed(3)
n = 10000

dW = np.zeros(n)
dW[0] = 0
for i in range(n-1):
    dW[i+1] = np.random.normal(0, np.sqrt(0.1))
    dV_tilde = np.random.normal(0, np.sqrt(0.1))
    
W = np.cumsum(dW)

from matplotlib import pyplot as plt

plt.plot(W)