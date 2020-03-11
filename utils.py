# -*- coding:utf-8 -*-
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import mahotas
import os

def color_reverse(img):
    height, width = img.shape[:2]
    for j in range(1, width, 1): # 颜色反转
        for i in range(1, height, 1):
            color = img[i,j]
            img[i, j] = 255 - color
    return img


def simple_threshold(gray):
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    return binary


def custom_threshold(gray):
    # 用户自己计算阈值
    # h, w =gray.shape[:2]
    # m = np.reshape(gray, [1,w*h])
    # mean = m.sum()/(w*h)
    # ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    # return binary
    height, width = gray.shape[:2]
    mascar = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.rectangle(mascar, (0, 0), (width, height), 255, -1)
    gris = cv2.GaussianBlur(gray, (3, 3), 0)
    T1 = mahotas.thresholding.otsu(gris)
    clahe = cv2.createCLAHE(clipLimit=1.0)
    grises = clahe.apply(gris)
    T2 = mahotas.thresholding.otsu(grises)
    T = (T2 + T1 + 5) / 2
    # THRESHOLD--------------------------------------------------------------------------------------------------------------
    for k in range(0, height, 1):
        for z in range(0, width, 1):
            color = grises[k, z]
            if color > T:
                grises[k, z] = 255
            else:
                grises[k, z] = 0
    # MASCARA FOR ROI--------------------------------------------------------------------------------------------------------
    mascara = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.rectangle(mascara, (0, 0), (width, height), 255, -1)
    t = cv2.bitwise_and(grises, grises, mask=mascara)
    return t


def threshold(image, is_gray=True):
    gray = image
    if not is_gray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    mascar = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mascar, (0, 0), (width, height), 255, -1)
    gris = cv2.GaussianBlur(gray, (3, 3), 0)
    T1 = mahotas.thresholding.otsu(gris)
    clahe = cv2.createCLAHE(clipLimit=1.0)
    grises = clahe.apply(gris)
    T2 = mahotas.thresholding.otsu(grises)
    T = (T2 + T1 + 5) / 2
    # THRESHOLD--------------------------------------------------------------------------------------------------------------
    for k in range(0, height, 1):
        for z in range(0, width, 1):
            color = grises[k, z]
            if color > T:
                grises[k, z] = 255
            else:
                grises[k, z] = 0
    # MASCARA FOR ROI--------------------------------------------------------------------------------------------------------
    mascara = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mascara, (0, 0), (width, height), 255, -1)
    t = cv2.bitwise_and(grises, grises, mask=mascara)
    return t


def correct_skew(image, delta=1, limit=5, is_gray=False):
    # 倾斜矫正
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    if is_gray:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


if __name__ == '__main__':
    img = cv2.imread('./test0310/2.png', cv2.IMREAD_GRAYSCALE)
    thresh = custom_threshold(img)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    # root = './test0309-samples'
    # dirList = os.listdir(root)
    # i = 1
    # for imgName in dirList:
    #     # 解决文件夹中有 .DS_STORE的情况
    #     path = os.path.join(root, imgName)
    #     if imgName.startswith('.') or os.path.isdir(path):
    #         continue
    #     print 'dir: {}'.format(imgName)
    #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #     tmp = custom_threshold(img)
    #     cv2.imwrite('./test0309-samples-threshold/{}'.format(imgName), tmp)
    # print 'finish'