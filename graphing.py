#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:42:47 2020

@author: lfi
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

filename1 = '/hdd3/Automap/experiments/train/checkpoint/loss_training.p'
val = np.load(open(filename1,'rb'))
plt.plot(val[0,:], label='Training')
plt.plot(val[1,:], label='Test')
plt.title('AUTOMAP')
plt.legend()
print(val[0,-1])
print(val[1,-1])


