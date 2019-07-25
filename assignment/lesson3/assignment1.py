
import numpy as np
import random
# 生成多个样本，每个样本都有x值的矩阵和y的矩阵，先随机生成w,b，设置训练次数，每次训练遍历所有样本，根据每个样本算出新的w和b，进入下次运

def getSamplesData(w,b,nums = 10):
    data = []
    for i in range(nums):
        noise = np.random.randn(1) * 0.01
        x0 = np.random.randn(1,w.shape[0])
        y0 = np.dot(x0,w)+b+noise
        a = [x0,y0]
        data.append(a)
    return data

def train(w,b,data,learnRate):
    for x0,y0 in data:
        y = np.dot(x0,w) + b
        dif = y - y0
        w_g = dif*x0.T
        b_g = dif[0]
        w -= w_g*learnRate
        b -= b_g*learnRate
        loss = np.square(dif)*0.5
    return [w,b,loss[0][0]]

def linearRegresion(data):
    w = np.random.randn(data[0][0].shape[1],1)
    b = np.random.randn(1)
    for i in range(10000):
        w,b,loss = train(w,b,data,0.01)
        if i%100 == 0 :
            print(loss)
    return [w,b]



if __name__ == '__main__':
    w = np.array([[2],[3],[4],[5]])
    b = np.array([1])
    data = getSamplesData(w,b)
    w0,b0 = linearRegresion(data)
    print(w0)
    print(b0)



