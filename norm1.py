import math
import iisignature
import matplotlib
import numpy

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

n=10000
T= 1.
times = np.linspace(0., T, n)
dt = times[1]-times[0]
dB = np.sqrt(dt)*np.random.normal(size=(n-1,))
dB1 = np.sqrt((dt))*np.random.normal(size=(n-1,))
B0 = np.zeros(shape=(1,))
B=np.concatenate((B0, np.cumsum(dB))).reshape((-1,1))
B1=np.concatenate((B0, np.cumsum(dB1))).reshape((-1,1))

BPath = np.append(B,B1,axis=1)


K=[]

x = np.linspace(2,12,11)

for i in x:
    sig1 =iisignature.sig(BPath, int(i-1))
    sig2 =iisignature.sig(BPath, int(i))
    siglevel = np.array([j for j in sig2 if j not in sig1])
    a = math.gamma((i/ 2) + 1) * (siglevel ** (i))
    norm = np.linalg.norm(a)
    K.append(math.pow(norm, float(2)/ float(i)))

plt.plot(x,K)

plt.show()



























