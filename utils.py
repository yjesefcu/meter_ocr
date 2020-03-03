# -*- coding:utf-8 -*-
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter


def color_reverse(img):
    height, width = img.shape[:2]
    for j in range(1, width, 1): # 颜色反转
        for i in range(1, height, 1):
            color = img[i,j]
            img[i, j] = 255 - color


def custom_threshold(gray):
    # 用户自己计算阈值
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    return binary


def correct_skew(image, delta=1, limit=5):
    # 倾斜矫正
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


if __name__ == '__main__':
    # img = cv2.imread('./dianbiao/Screenshot_20200221_130816.jpg')
    # angle, rotated = correct_skew(img)
    # print(angle)
    # cv2.imshow('rotated', rotated)
    # cv2.waitKey()
    f = open('./data/general_responses.data', mode='r')
    lines = f.readlines()
    f.close()