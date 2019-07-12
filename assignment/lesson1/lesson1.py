
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

#把图像放进去就可以了
def combineProcessImage(img):
    #   crop
    cropRand = random.randint(5,10)
    width = int(img.shape[0] * (cropRand/10.0))
    height = int(img.shape[1] * (cropRand/10.0))
    img_crop = img[0:width, 0:height]

    # change color
    B, G, R = cv2.split(img_crop)
    B1 = changeChannelColor(B,img_crop)
    G1 = changeChannelColor(G,img_crop)
    R1 = changeChannelColor(R,img_crop)
    img_merge = cv2.merge((B1,G1,R1))

    #gamma
    gammaRand = random.randint(1, 5)
    invGamma = 1.0 / gammaRand
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    img_gamma = cv2.LUT(img_merge, table)

    #rotation scala
    rotationRand = random.randint(30, 360)
    rotationScaleRand = random.randint(5,10)
    M = cv2.getRotationMatrix2D((img_gamma.shape[1] / 2, img_gamma.shape[0] / 2), rotationRand, rotationScaleRand/10)
    img_rotate = cv2.warpAffine(img_gamma, M, (img_gamma.shape[1], img_gamma.shape[0]))

    # Affine Transform
    rows, cols, ch = img_rotate.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    img_dst = cv2.warpAffine(img_rotate, M, (cols, rows))

    img_warp = random_warp(img_rotate)
    return img_warp

def changeChannelColor(cn,img):
    b_rand = random.randint(-75, 75)

    B = cn
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    return B

# perspective transform
def random_warp(img):
    row = img.shape[0]
    col = img.shape[1]
    height, width, channels = img.shape

    # warp:
    random_margin = random.randint(50, 100)
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp

#主程序 请自己配置图片路径！！！
img = cv2.imread("/Users/litao/Documents/project/cv/Lenna_raw.jpg")
newImg = combineProcessImage(img)
cv2.imshow('lenna_new',newImg)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()