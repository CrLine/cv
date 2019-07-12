


import cv2
import numpy as np


def m_filter(x,y,step,im):
    sum_s = []
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s.append(im[x+k][y+m])
    sum_s.sort()
    return sum_s[(int(step*step/2)+1)]

def mean_filter(x,y,step,im):
    sum_s = 0
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s += im[x+k][y+m]/(step*step)
    return sum_s

def meanBlur(img, kernel, padding_way):
    w = kernel.shape[0]
    h = kernel.shape[1]
    w2 = int(w/2)
    newimage = cv2.copyMakeBorder(img,w2,w2,w2,w2,cv2.BORDER_REPLICATE)
    medianImage = newimage.copy()
    if padding_way == 'ZERO':
        newimage = cv2.copyMakeBorder(img, w2, w2, w2, w2, cv2.BORDER_CONSTANT,0)
    for i in range(int(w2), newimage.shape[0] - int(w2)):
        for j in range(int(w2), newimage.shape[1] - int(w2)):
            medianImage[i][j] = mean_filter(i, j, w,newimage)
    return  medianImage

def medianBlur(img, kernel, padding_way):
    medianImage = img.copy()
    w = kernel.shape[0]
    h = kernel.shape[1]
    w2 = int(w/2)
    newimage = cv2.copyMakeBorder(img,w2,w2,w2,w2,cv2.BORDER_REPLICATE)
    if padding_way == 'ZERO':
        newimage = cv2.copyMakeBorder(img, w2, w2, w2, w2, cv2.BORDER_CONSTANT,0)
    for i in range(int(w2), newimage.shape[0] - int(w2*2)):
        for j in range(int(w2), newimage.shape[1] - int(w2*2)):
            medianImage[i-w2][j-w2] = m_filter(i, j, w,newimage)
    return  medianImage

def main() :

    fn = "/Users/litao/Documents/project/cv/Lenna_raw.jpg"
    img = cv2.imread(fn)
    B, G, R = cv2.split(img)

    # 加上椒盐噪声
    # 灰阶范围
    w = img.shape[1]
    h = img.shape[0]
    im = np.array(B)
    # print(im)
    # 噪声点数量
    noisecount = 5000
    for k in range(0, noisecount):
        xi = int(np.random.uniform(0, im.shape[1]))
        xj = int(np.random.uniform(0, im.shape[0]))
        im[xj, xi] = 255

    cv2.imshow('yuantu', B)
    cv2.imshow('jiaoyan', im)

    kernel = np.ones((3,3))

    medianImg = medianBlur(im,kernel,"ZERO")

    cv2.imshow('middle',medianImg)
    #cv2.imshow('averate',im_copy_mea)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

main()




