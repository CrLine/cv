
import numpy as np
import random
from numpy import *

def logicRegresion(data,w,learate):
    for x0,y0 in data:
        y = np.dot(x0,w)
        y = sigmoid(y)
        dif = y[0]-y0
        #print(dif)
        w_g = dif*x0.T
        #print(w_g)
        w -= w_g*learate
    return w

def train(data):
    w = np.random.randn(data[0][0].shape[1],1)
    for i in range(10000):
        w = logicRegresion(data,w,0.01)
    return w


def sigmoid(inx):
    return 1/(1+exp(-inx))

def getSamplesData(w,nums = 10):
    data = []
    for i in range(nums):
        noise = np.random.randn(1)*0.01
        x0 = np.random.randn(1,w.shape[0])
        y0 = np.dot(x0,w)
        #print(y0)
        y0 = sigmoid(y0)
        #print(y0)
        a = [x0,y0[0]]
        data.append(a)
    return data


if __name__ == '__main__':
   w = np.array([[2],[3],[4],[5]])
   data = getSamplesData(w)
   w0 = train(data)
   print(w0)

