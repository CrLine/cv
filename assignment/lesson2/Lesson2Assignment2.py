import cv2
import numpy as np

# 关于RANSAC
# 1 数据是由局内点组成
# 2 局外点是不能适应模型的数据
# 3 除此之外都是噪声
# 1 随机假设一个小组局内点为初始值，然后用此局内点拟合一个模型，此模型适应于假设的局内点，所有的未知参数都能从假设的局内点计算出
# 2 用1中得到的模型去测试所有的其他数据，如果某个点适用于估计的模型，认为它也是局内点，将局内点扩充
# 3 如果足够多的点被归类为假设的局内点，那么估计的模型就足够合理
# 4 然后，用所有假设的局内点去重新估计模型，因为此模型仅仅是在出事的假设的局内点估计的，后续由扩充后，需要更新。
# 5 最后，通过估计局内点与模型的错误率来评估模型

#伪代码
# n 适用于模型的最少数据个数
# k 算法的迭代次数
# t 用于决定数据是否适合于模型的阀值
# d 判断模型是否适用于数据集的数据数目
# iterations = 0
# best_model = null
# best_consensus_set = null
# best_error = 无穷大
# while (iterations < k)
#     maybe_inliers = 从数据集中随机选择n个点
#     maybe_model = 适合于maybe_inliers的模型参数
#     consensus_set = maybe_inliers
#     for(每个数据集中不属于maybe_inliers的点):
#         if(如果点适合于maybe_model,且错误小于t):
#             将点添加到consensus_set
#     if(consensus_set中的元素大于d)
#         已经找到了好的模型，现在测试该模型到底由多好
#     better_model = 适合于consensus_set中所有点的模型参数
#     this_error = better_model究竟如何适应这些点的度量
#     if(this_error < best_error)
#         我们发现了比以前好的模型，保存该模型直到跟好的模型出现
#     best_model = better_model
#     best_consensus_set = consensus_set
#     best_error = this_error
#     增加迭代次数
#     返回 best_model,best_consensus_set,best_error

# 2) 以灰度图的形式读入图片

min_match_count = 10
psd_img_1 = cv2.imread('/Users/litao/Documents/project/cv/code/lesson2/long1.jpg', cv2.IMREAD_GRAYSCALE)
psd_img_2 = cv2.imread('/Users/litao/Documents/project/cv/code/lesson2/long2.jpg', cv2.IMREAD_GRAYSCALE)

# 3) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()

psd_kp1, psd_des1 = sift.detectAndCompute(psd_img_1, None)
psd_kp2, psd_des2 = sift.detectAndCompute(psd_img_2, None)

# 4) Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)
goodMatch = []
for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50*n.distance:
        goodMatch.append([m])
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
#print(goodMatch[:20])

'''
设置只有存在10个以上匹配时，采取查找目标 min_match_count=10，否则显示特征点匹配不了
如果找到了足够的匹配，就提取两幅图像中匹配点的坐标，把它们传入到函数中做变换
'''

if len(goodMatch) > min_match_count:
    # 获取关键点的坐标
    src_pts = np.float32([psd_kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    dst_pts = np.float32([psd_kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    # 第三个参数 Method used to computed a homography matrix.
    #  The following methods are possible: #0 - a regular method using all the points
    # CV_RANSAC - RANSAC-based robust method
    # CV_LMEDS - Least-Median robust method
    # 第四个参数取值范围在 1 到 10  绝一个点对的 值。原图像的点经 变换后点与目标图像上对应点的 差 #    差就 为是 outlier
    #  回值中 M 为变换矩 。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # 获取原图像的高和宽
    h, w = psd_img_1.shape
    # 使用得到的变换矩阵对原图想的四个变换获得在目标图像上的坐标
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # 将原图像转换为灰度图
    img2 = cv2.polylines(psd_img_2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print('Not enough matches are found - %d/%d' % (len(goodMatch), min_match_count))
    matchesMask = None

# 最后在绘制inliers，如果能成功找到目标图像的话或者匹配关键点失败
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)



img_out = cv2.drawMatchesKnn(psd_img_1, psd_kp1, psd_img_2, psd_kp2, goodMatch[:], None, **draw_params)

cv2.imshow('image', img_out)#展示图片
cv2.waitKey(0)#等待按键按下
cv2.destroyAllWindows()#清除所有窗口
