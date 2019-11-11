import numpy as np
import cv2

image1 = cv2.imread('1.jpg')
image2 = cv2.imread('2.jpg')
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# sift寻找关键点以及对应的关键向量
kp1, kp2 = {}, {}
kp1['kp'], kp1['des'] = cv2.xfeatures2d.SIFT_create().detectAndCompute(img1, None)
kp2['kp'], kp2['des'] = cv2.xfeatures2d.SIFT_create().detectAndCompute(img2, None)
# 计算描述子之间的相似度
matches = cv2.BFMatcher_create().knnMatch(kp1['des'], kp2['des'], k=2)
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        good.append((m.trainIdx, m.queryIdx))

if len(good) > 4:
    points1 = kp1['kp']
    points2 = kp2['kp']
    matched_kp1 = np.float32([points1[i].pt for (_, i) in good])
    matched_kp2 = np.float32([points2[i].pt for (i, _) in good])
    # 求homomatrix,排除异常值
    homo_matrix,_ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 4)

# 画图
h1, w1 = image1.shape[0], image1.shape[1]
h2, w2 = image2.shape[0], image2.shape[1]
rect1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape((4, 1, 2))
rect2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape((4, 1, 2))
trans_rect1 = cv2.perspectiveTransform(rect1, homo_matrix)
total_rect = np.concatenate((rect2, trans_rect1), axis=0)  # 数组拼接
min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
max_x, max_y = np.int32(total_rect.max(axis=0).ravel())
shift_to_zero_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
trans_img1 = cv2.warpPerspective(image1, shift_to_zero_matrix.dot(homo_matrix), (max_x - min_x, max_y - min_y))
trans_img1[-min_y:h2 - min_y, -min_x:w2 - min_x] = image2

cv2.imwrite('output.jpg',trans_img1)


