# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import mahotas

#全局阈值
def threshold_demo(gray):
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    cv.imshow("binary0", binary)
    return binary


#局部阈值
def local_threshold(gray):
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    # binary =  cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
    binary =  cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 255, 10)
    cv.namedWindow("binary1", cv.WINDOW_NORMAL)
    cv.imshow("binary1", binary)
    return binary


#用户自己计算阈值
def custom_threshold(gray):
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.namedWindow("binary2", cv.WINDOW_NORMAL)
    cv.imshow("binary2", binary)
    return binary



if __name__ == '__main__':
    src = cv.imread('./images/img1.jpg')
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.namedWindow('input_image', cv.WINDOW_NORMAL) # 设置为WINDOW_NORMAL可以任意缩放
    cv.imshow('input_image', src)
    threshold_demo(gray)
    local_threshold(gray)
    custom_threshold(gray)
    binary = cv.bitwise_and(gray, gray)
    cv.imshow('binary3', binary)
    cv.waitKey(0)
    cv.destroyAllWindows()