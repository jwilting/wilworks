import numpy as np
import sys
import pylab as plt

   

N = 10000
x = np.empty((N, 4))
for t in range(N):
    m = np.random.uniform()
    h = np.random.uniform()
    n = np.random.uniform()
    print m, h, n
    x[t,0] = m
    x[t, 1] = h
    x[t, 2] = n
    x[t, 3] = 10 * m + 5 * h**2 + n * h

np.savetxt('Data/simple_train.csv', x, delimiter=',')
