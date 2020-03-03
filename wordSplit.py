# -*- coding:utf-8 -*-
import cv2
import numpy as np
import utils


def custom_threshold(gray):
    # 用户自己计算阈值
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    return binary


# 字符分割
def colorReverse(img):
    img_info = img.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    dst = np.zeros((image_height, image_weight, 1), np.uint8)
    for i in range(image_height):
        for j in range(image_weight):
            grayPixel = img[i][j]
            dst[i][j] = 255 - grayPixel
    cv2.imshow('gary', dst)
    return dst


def column_split(img):
    # 垂直分割
    height, width = img.shape[:2]
    v = [0] * width
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == 0:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    wordPos = [] # 记录每个字符的起止点位 [(start,end),(start,end)]
    wordWidth = [] # 每个字符所占宽度
    maxCount = max(v) #
    startIndex = -1
    # 通过黑色占比去掉一些燥点
    for i in range(0, len(v)):
        if (height - v[i]) / float(maxCount) < 1 / float(7): # 将黑色点数占比不到1/7的去除，避免一些燥点被算进去
            v[i] = height
            if startIndex >= 0:
                wordPos.append((startIndex, i - 1))
                wordWidth.append(i - startIndex)
                startIndex = -1
        else:
            if startIndex < 0:
                startIndex = i
    if startIndex >= 0: # 如果最后一列仍是字符
        wordPos.append((startIndex, i - 1))
        wordWidth.append(i - startIndex)
    if len(wordWidth) == 0:
        return []
    # 根据wordPos每个字符的宽度，判断是否有效字符
    actualWordsPos = [] # 最终实际的字符的x坐标
    medianWidth = np.median(wordWidth) # 宽度的中位数
    for pos in wordPos:
        w = pos[1] - pos[0]
        if w > medianWidth/2.5: # 如果宽度小于宽度中位数的一半，则不认为是字符
            actualWordsPos.append(pos)
    if len(actualWordsPos) == 0:
        return []
    # 排除掉间隔很近的字符
    cv2.imshow('columnSplit', img)
    # 创建空白图片，绘制垂直投影图
    emptyImage = np.zeros((height, width, 3), np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            b = (255, 255, 255)
            emptyImage[y, x] = b
    cv2.imshow('chuizhi', emptyImage)
    # 将字符取出
    return actualWordsPos


def line_split(word, whole_split=False, show_window=None):
    # 对单个字符进行行分割, whole_split:表示是整张图片的分割，还是单个字符的分割
    # 水平投影  #统计每一行的黑点数
    height, width = word.shape[:2]
    v = [0] * height
    a = 0
    emptyImage1 = np.zeros((height, width, 3), np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            if word[y, x] == 0:
                a = a + 1
            else:
                continue
        v[y] = a
        a = 0
    # 将黑色点太少的去除
    wordPos = []  # 记录每个字符的起止点位 [(start,end),(start,end)]
    wordWidth = []  # 每个字符所占宽度
    maxCount = max(v)  #
    startIndex = -1
    # 通过黑色占比去掉一些燥点
    for i in range(0, len(v)):
        if (whole_split and v[i] < (width / 20)) or \
                ((width - v[i]) / float(maxCount)) < (1 / float(10)):  # 将黑色点数占比不到1/10的去除，避免一些燥点被算进去
            v[i] = width
            if startIndex >= 0:
                wordPos.append((startIndex, i - 1))
                wordWidth.append(i - startIndex)
                startIndex = -1
        else:
            if startIndex < 0:
                startIndex = i
    if startIndex >= 0:  # 如果最后一列仍是字符
        wordPos.append((startIndex, i - 1))
        wordWidth.append(i - startIndex)
    maxIndex = np.argmax(wordWidth) # 宽度最大的占比，作为最终的字符位置
    # 绘制水平投影图
    for y in range(0, height):
        for x in range(0, v[y]):
            b = (255, 255, 255)
            emptyImage1[y, x] = b
    if show_window:
        cv2.imshow(show_window, emptyImage1)
    return wordPos[maxIndex]


def img_to_words(img):
    # 将图片分割成字符
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # (_, thresh) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    thresh = custom_threshold(gray) # 二值化
    # 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 形态学处理:定义矩形结构
    closed = cv2.erode(thresh, kernel)  # 腐蚀
    # # 开运算：先腐蚀再膨胀
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)
    cv2.imshow('thresh', thresh)
    # closed = colorReverse(closed) # 颜色反转
    cv2.imshow('closed', closed)
    height, width = closed.shape[:2]
    # 先进行一次行分割
    (h0, h1) = line_split(closed, whole_split=True, show_window='shuiping')
    closed = closed[h0:h1, 0:width]
    # 进行列分割
    xWords = column_split(closed) # 垂直的字符位置
    if len(xWords) == 0:
        print 'column_split return empty'
        return []
    wordRects = []  # 最终的字符的(x,y,w,h)
    for i in range(0, len(xWords)):
        xPos = xWords[i]
        # cv2.rectangle(img, (xPos[0], 0), (xPos[1], height), (0, 0, 255), 2) # 用矩形显示列分割
        # 对每个字符进行水平分割
        wordRect = closed[0:height, xPos[0]:xPos[1]]
        yPos = line_split(wordRect, show_window=False)
        x = xPos[0]
        y = yPos[0] + h0
        w = xPos[1] - xPos[0]
        h = yPos[1] - yPos[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 用矩形显示最终字符
        wordRects.append((x, y, w, h))

    cv2.imshow('words', img)
    return wordRects # 返回每个字符的(x,y,w,h)


if __name__ == '__main__':
    img = cv2.imread('./area/44.png')
    angle, img = utils.correct_skew(img)
    cv2.imshow('skew', img)
    img_to_words(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()