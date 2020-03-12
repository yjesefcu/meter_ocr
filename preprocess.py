# -*- coding:utf-8 -*-
# 图像预处理
import cv2
import os
import utils
import numpy as np


def get_black_mat(img, show=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 0, 0])
    high_hsv = np.array([180, 255, 46])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    if show:
        cv2.imshow('black area', mask)
    return mask


def get_red_mat(img, show=False): # 取图像的红色所在的点，返回一个矩阵，红色值用255表示，其他用0表示
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    low_hsv = np.array([156, 43, 46])  # 偏粉红
    high_hsv = np.array([180, 255, 255])  # np.array([10, 255, 255])
    mask2 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    # 取两部分的合集
    mask = mask1 + mask2
    if show:
        cv2.imshow('red area', mask)
    return mask


def get_meter_red_area(img, show=False): # 获取电表的红色区域，返回红色区域的起止x坐标 (x0, x1)
    mat = get_red_mat(img, show)
    # 对mat进行膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 形态学处理:定义矩形结构
    erode = cv2.erode(mat, kernel, iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=1)  # 膨胀
    if show:
        cv2.imshow('red dilated', dilated)
    # 通过垂直投影，得到红色的x坐标
    v, groups, widths = column_shadow(dilated, color=255)
    if len(groups) == 0:
        return None
    m = 0
    max_group = None
    height, width = img.shape[:2]
    for g in groups:
        diff = g[1] - g[0]
        xs = v[g[0]:g[1] + 1]
        xsmax = max(xs)
        if xsmax < height/5.:
            continue
        if m < diff:
            m = diff
            max_group = (g[0], g[1])
    if show and max_group:
        cv2.rectangle(img, (max_group[0], 0), (max_group[1], img.shape[0]), (0, 0, 255), 2)  # 用矩形显示最终字符
        cv2.imshow('red area rect', img)
    return max_group


def convert_red_to_black(img, show=False): # 将图片中的红色转成黑色
    mask = get_red_mat(img, show)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    height, width = img.shape[:2]
    copy = img.copy()
    for y in range(0, height):
        for x in range(0, width):
            if mask[y][x] == 255:
                copy[y][x] = (0, 0, 0)
    if show:
        cv2.imshow('black', copy)
    return copy


def column_shadow(thresh, color=255, threshold=0, show=False): # 需传入二值化图片
    # 垂直投影，只返回投影后的白色(color=255)或黑色(color=0)点位个数
    # return: (v, groups), v为投影后每一行color的个数，groups为将相连的个数不为0的区域进行统计后得到的组的起止点位
    height, width = thresh.shape[:2]
    v = [0] * width
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if thresh[y, x] == color:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    groups = []
    start_index = -1
    group_height = []
    for i in range(0, width):
        if v[i] > threshold and start_index < 0:
            start_index = i
        elif (v[i] <= threshold and start_index >= 0) or (i == width - 1 and start_index >= 0):
            groups.append((start_index, i - 1))
            group_height.append(i - start_index)
            start_index = -1
    emptyImage1 = np.full((height, width, 3), 255-color, dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            b = (color, color, color)
            emptyImage1[y, x] = b
    if show:
        cv2.imshow('line shadow', emptyImage1)
    return v, groups, group_height


def line_shadow(thresh, color=255, threshold=0, show=False): # 需传入二值化图片
    # 水平投影，只返回投影后的白色(color=255)或黑色(color=0)点位个数
    # return: (v, groups), v为投影后每一行color的个数，groups为将相连的个数不为0的区域进行统计后得到的组的起止点位
    height, width = thresh.shape[:2]
    v = [0] * height
    a = 0
    # 统计水平方向的白色个数
    for y in range(0, height):
        for x in range(0, width):
            if thresh[y, x] == color:
                a = a + 1
            else:
                continue
        v[y] = a
        a = 0
    groups = []
    start_index = -1
    group_height = []
    for i in range(0, height):
        if v[i] > threshold and start_index < 0:
            start_index = i
        elif (v[i] <= threshold and start_index >= 0) or (i == height - 1 and start_index >= 0):
            groups.append((start_index, i - 1))
            group_height.append(i - start_index)
            start_index = -1
    emptyImage1 = np.full((height, width, 3), 255-color, dtype=np.uint8)
    for y in range(0, height):
        for x in range(0, v[y]):
            b = (color, color, color)
            emptyImage1[y, x] = b
    if show:
        cv2.imshow('line shadow', emptyImage1)
    return v, groups, group_height


def rect_boundary(grayImg, show=False):
    # 数字区域定位
    # thresh = binary.local_threshold(grayImg) # 二值化
    # thresh = utils.custom_threshold(grayImg)
    thresh = utils.simple_threshold(grayImg)
    thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    if show:
        cv2.imshow('thresh', thresh)
    # 取最中间的一段图像做水平投影，取出黑色占比最大的一段
    height, width = grayImg.shape[:2]
    middle_img = thresh[0:height, (width/2-width/8):(width/2+width/8)]
    if show:
        cv2.imshow('middle', middle_img)
    # 对截取的图片进行水平黑色投影，取出黑色占比最多的部分
    line_colors, groups, widths = line_shadow(middle_img, color=0, show=show) # 对中间图片进行水平黑色投影
    groups = []
    start_index = -1
    group_height = []
    for i in range(0, height):
        if line_colors[i] > width / 20 and start_index < 0:
            start_index = i
        elif (line_colors[i] <= width /20 and start_index >= 0) or (i == height - 1 and start_index >= 0):
            groups.append((start_index, i - 1))
            group_height.append(i - start_index)
            start_index = -1
    max_index = np.argmax(group_height)
    posY = groups[max_index]
    rect_vertical = thresh[posY[0]:posY[1], 0:width] # 取出水平的中间部分
    if show:
        cv2.imshow('vertical rect', rect_vertical)

    # 按垂直投影，去掉两端非黑色的部分
    column_colors, groups, widths = column_shadow(rect_vertical, color=0, show=show)
    groups = []
    start_index = -1
    group_height = []
    for i in range(0, width):
        if column_colors[i] > 0 and start_index < 0:
            start_index = i
        elif (column_colors[i] == 0 and start_index >= 0) or (i == width - 1 and start_index >= 0):
            groups.append((start_index, i - 1))
            group_height.append(i - start_index)
            start_index = -1
    max_index = np.argmax(group_height)
    posX = groups[max_index]
    # rect_horizontal = rect_vertical[0: h, posX[0]:posX[1]]
    if show:
        rect = grayImg[posY[0]:posY[1], posX[0]:posX[1]]
        cv2.imshow('rect', rect)
    return (posX[0], posY[0], posX[1]-posX[0], posY[1]-posY[0])


def rect_boundary2(img, show=False): # 通过HSV获取黑色部分
    converted = convert_red_to_black(img, show)
    mat = get_black_mat(converted, show=show)
    angle, rotated = utils.correct_skew(mat, is_gray=True)
    if show:
        cv2.imshow('rotated', rotated)
    # 水平投影
    v, groups, counts = line_shadow(rotated, color=255)


# 图像预处理
if __name__ == '__main__':
    img = cv2.imread('./test0312/17.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold = utils.custom_threshold(gray)
    # cv2.imshow('threshold', threshold)
    #
    # """
    # 提取图中的红色部分
    # """
    angle, img = utils.correct_skew(img, is_gray=False)
    get_red_mat(img, True)
    x0, x1 = get_meter_red_area(img, True)
    cv2.rectangle(img, (x0, 0), (x1, img.shape[0]), (255, 0, 0), 3)
    cv2.imshow('red', img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold = utils.custom_threshold(gray)
    # cv2.imshow('threshold1', threshold)
    # cv2.imshow('origin2', img)

    # angle, rotated = utils.correct_skew(gray, is_gray=True)
    # cv2.imshow('rotated', rotated)
    # (x, y, w, h) = rect_boundary(rotated, show=True)

    # # rect = rotated[y:y+h, x:x+2]
    # # 将图片分成上下各十份进行二值化
    # height, width = img.shape[:2]
    # xs = np.array_split(np.arange(width), 6)
    # ys = np.array_split(np.arange(height), 2)
    # threshold = np.zeros(img.shape[:2], dtype="uint8")
    # for i in range(0, 6):
    #     for j in range(0, 2):
    #         x0 = xs[i][0]
    #         x1 = xs[i][-1]
    #         y0 = ys[j][0]
    #         y1 = ys[j][-1]
    #         t = utils.custom_threshold(gray[y0: y1, x0: x1])
    #         # t = utils.simple_threshold(gray[y0: y1, x0: x1])
    #         for x in range(0, x1-x0):
    #             for y in range(0, y1-y0):
    #                 threshold[y+y0][x+x0] = t[y][x]
    # cv2.imshow('threshold2', threshold)

    # rect_boundary2(img, show=True)

    cv2.waitKey(0)

