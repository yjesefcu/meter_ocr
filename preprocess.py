# -*- coding:utf-8 -*-
# 图像预处理
import cv2
import os
import utils
import numpy as np
import binary


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


def get_meter_red_area(img, show=False): # 获取电表的红色区域
    mat = get_red_mat(img, show)
    # 对mat进行膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 形态学处理:定义矩形结构
    dilated = cv2.dilate(mat, kernel, iterations=1)  # 膨胀
    dilated = cv2.erode(dilated, kernel, iterations=1)
    # 通过垂直投影，得到红色的x坐标
    v, groups = _column_shadow(dilated, color=255)
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


def _column_shadow(thresh, color=255, show=False): # 需传入二值化图片
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
        if v[i] > 0 and start_index < 0:
            start_index = i
        elif (v[i] == 0 and start_index >= 0) or (i == width - 1 and start_index >= 0):
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
    return v, groups


def _line_shadow(thresh, color=255, show=False): # 需传入二值化图片
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
        if v[i] > 0 and start_index < 0:
            start_index = i
        elif (v[i] == 0 and start_index >= 0) or (i == height - 1 and start_index >= 0):
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
    return v, groups


def rect_boundary(grayImg, show=False):
    # 数字区域定位
    # thresh = binary.local_threshold(grayImg) # 二值化
    # thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    thresh = binary.custom_threshold(grayImg)
    if show:
        cv2.imshow('thresh', thresh)
    # 取最中间的一段图像做水平投影，取出黑色占比最大的一段
    height, width = grayImg.shape[:2]
    middle_img = thresh[0:height, (width/2-width/10):(width/2+width/10)]
    if show:
        cv2.imshow('middle', middle_img)
    # 对截取的图片进行水平黑色投影，取出黑色占比最多的部分
    line_colors, groups = _line_shadow(middle_img, color=0, show=show) # 对中间图片进行水平黑色投影
    groups = []
    start_index = -1
    group_height = []
    for i in range(0, height):
        if line_colors[i] > 0 and start_index < 0:
            start_index = i
        elif (line_colors[i] == 0 and start_index >= 0) or (i == height - 1 and start_index >= 0):
            groups.append((start_index, i - 1))
            group_height.append(i - start_index)
            start_index = -1
    max_index = np.argmax(group_height)
    posY = groups[max_index]
    rect_vertical = thresh[posY[0]:posY[1], 0:width] # 取出水平的中间部分
    if show:
        cv2.imshow('vertical rect', rect_vertical)

    # 按垂直投影，去掉两端非黑色的部分
    column_colors, groups = _column_shadow(rect_vertical, color=0, show=show)
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


# 图像预处理
if __name__ == '__main__':
    img = cv2.imread('./area/62.png')
    cv2.imshow('origin', img)
    """
    提取图中的红色部分
    """
    # get_meter_red_area(img, show=True)

    # show = False
    # root = './train-source1'
    # dirList = os.listdir(root)
    # for imgName in dirList:
    #     # 解决文件夹中有 .DS_STORE的情况
    #     path = os.path.join(root, imgName)
    #     if imgName.startswith('.') or os.path.isdir(path):
    #         continue
    #     img = cv2.imread('{}/{}'.format(root, imgName))
    #     x = get_meter_red_area(img, show=True)
    #     if x:
    #         cv2.rectangle(img, (x[0], 0), (x[1], img.shape[0]), (0, 0, 255), 2)  # 用矩形显示最终字符
    #     cv2.imwrite('./red/{}'.format(imgName), img)
    # print 'finish'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle, img = utils.correct_skew(gray, is_gray=True)
    rect_boundary(img, show=True)

    # img = utils.custom_threshold(img)
    # cv2.imshow("thresh", img)
    # kernel = cv2.getstructuringelement(cv2.morph_rect, (8, 8))  # 形态学处理:定义矩形结构
    # img = cv2.erode(img, kernel, iterations=1)  # 腐蚀
    # cv2.imshow("erode", img)
    #
    # kernel = cv2.getstructuringelement(cv2.morph_rect, (10, 10))  # 形态学处理:定义矩形结构
    # img = cv2.dilate(img, kernel, iterations=1)
    # cv2.imshow("dilate", img)


    # img = cv2.imread('./area/44.png')
    # imghsv = cv2.cvtcolor(img, cv2.color_bgr2hsv)
    # cv2.imshow("image_hsv", imghsv)
    # # 分离hsv三通道
    # channels = cv2.split(imghsv)
    # cv2.imshow("image_h", channels[0])
    # cv2.imshow("image_s", channels[1])
    # cv2.imshow("image_v", channels[2])

    cv2.waitKey(0)

