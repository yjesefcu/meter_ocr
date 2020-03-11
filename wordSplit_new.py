# -*- coding:utf-8 -*-
# 图像预处理
import cv2
import os
import utils
import numpy as np
import preprocess


def rect_boundary2(img, show=False): # 通过HSV获取黑色部分
    converted = preprocess.convert_red_to_black(img, show)
    mat = preprocess.get_black_mat(converted, show=show)
    angle, rotated = utils.correct_skew(mat, is_gray=True)
    if show:
        cv2.imshow('rotated', rotated)
    # 水平投影
    v, groups, counts = preprocess.line_shadow(rotated, color=255)



# 图像预处理
if __name__ == '__main__':
    img = cv2.imread('./test0310/2.png')
    cv2.imshow('origin', img)

    rect_boundary2(img, show=True)

    cv2.waitKey(0)