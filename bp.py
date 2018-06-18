import numpy as np
import sys
import pylab as plt

def bp(m, h, L):
    x = np.empty(L)
    x[0] = np.random.poisson(h/(1. - m))
    for t in range(1, L):
        x[t] = np.random.poisson(m * x[t - 1]) + np.random.poisson(h)

    return x    

N = 1000
L = 10000
x = np.empty((N, L + 2))
for t in range(N):
    m = np.random.uniform()
    h = 10 * np.random.uniform()
    x[t,0] = m
    x[t, 1] = h
    x[t, 2:] = bp(m, h, L)

np.save('Data/bp_train.npy', x)
