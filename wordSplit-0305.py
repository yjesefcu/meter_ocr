# -*- coding:utf-8 -*-
import cv2
import numpy as np
import utils
import preprocess


def custom_threshold(gray):
    # 用户自己计算阈值
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    return binary


# 颜色反转
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


def column_shadow(img, color=255):
    # 垂直投影，只返回投影后的白色(color=255)或黑色(color=0)点位个数
    height, width = img.shape[:2]
    v = [0] * width
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == color:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    return v


def line_shadow(img, color=255):
    # 水平投影，只返回投影后的白色(color=255)或黑色(color=0)点位个数
    height, width = img.shape[:2]
    v = [0] * width
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == color:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    return v


def column_split(img, show=False):
    # 垂直分割
    v = column_shadow(img, color=255)
    wordPos = [] # 记录每个字符的起止点位 [(start,end),(start,end)]
    wordWidth = [] # 每个字符所占宽度
    maxCount = max(v) #
    startIndex = -1
    # 通过黑色占比去掉一些燥点
    for i in range(0, len(v)):
        if (v[i] / float(maxCount)) < (1 / float(20)): # 将黑色点数占比不到1/7的去除，避免一些燥点被算进去
            v[i] = 0
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
    # 创建空白图片，绘制垂直投影图
    height, width = img.shape[:2]
    emptyImage = np.full((height, width, 3), 0, np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            b = (255, 255, 255)
            emptyImage[y, x] = b
    if show:
        cv2.imshow('chuizhi', emptyImage)
    # 将字符取出
    return actualWordsPos


def line_split(word, whole_split=False, show_window=None):
    # 对单个字符进行行分割, whole_split:表示是整张图片的分割，还是单个字符的分割
    # 水平投影  #统计每一行的黑点数
    height, width = word.shape[:2]
    v = [0] * height
    a = 0
    # 统计水平方向的白色个数
    for y in range(0, height):
        for x in range(0, width):
            if word[y, x] == 255:
                a = a + 1
            else:
                continue
        v[y] = a
        a = 0
    # 将黑色点太少的去除
    wordPos = []  # 记录每个字符的起止点位 [(start,end),(start,end)]
    wordWidth = []  # 每个字符所占宽度
    maxCount = max(v)  #
    if maxCount == 0:
        return None
    startIndex = -1
    # 通过黑色占比去掉一些燥点
    for i in range(0, len(v)):
        if (whole_split and v[i] < (width / 20)) or \
                (v[i] / float(maxCount)) < (1 / float(10)):  # 将黑色点数占比不到1/10的去除，避免一些燥点被算进去
            v[i] = 0
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
    emptyImage1 = np.full((height, width, 3), 0, dtype=np.uint8)
    for y in range(0, height):
        for x in range(0, v[y]):
            b = (255, 255, 255)
            emptyImage1[y, x] = b
    if show_window:
        cv2.imshow(show_window, emptyImage1)
    return wordPos[maxIndex]


def rect_boundary(grayImg):
    # 数字区域定位
    thresh = utils.custom_threshold(grayImg) # 二值化
    # 取最中间的一段图像做水平投影，取出黑色占比最大的一段
    height, width = grayImg.shape[:2]
    middleImg = grayImg[0:height, (width/2-50):(width/2+50)]
    cv2.imshow('middle', middleImg)


def pre_process(img, show=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    img = utils.custom_threshold(img)
    if show:
        cv2.imshow("thresh", img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 形态学处理:定义矩形结构
    img = cv2.erode(img, kernel, iterations=2)  # 腐蚀
    if show:
        cv2.imshow("erode", img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 形态学处理:定义矩形结构
    img = cv2.dilate(img, kernel, iterations=2)
    if show:
        cv2.imshow("dilate", img)
    return img


# def after_process(words):
    # 对选取处理的字符进行后处理
    # 后处理的逻辑，对间隔太近


def img_to_words(img, show=False):
    # # 将图片分割成字符，1、识别出数字区；2、识别出红色区域；3、合并数字区和红色区域；4、二值化
    # 步骤1：识别出数字区域
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (x, y, w, h) = preprocess.rect_boundary(gray)
    digital_rect = img[y:y+h, x:x+2] # 取出数字区域
    # 步骤2：判断红色区域
    red_rect = preprocess.get_meter_red_area(digital_rect)


    # 图像预处理
    closed = pre_process(img, show)
    height, width = closed.shape[:2]
    # 先进行一次行分割
    # if show:
    #     (h0, h1) = line_split(closed, whole_split=True, show_window='shuiping')
    # else:
    #     (h0, h1) = line_split(closed, whole_split=True, show_window=None)
    closed = closed[y:(y+h), x:(x+w)]
    # 进行列分割
    xWords = column_split(closed, show) # 垂直的字符位置
    cv2.imshow('closed', closed)
    if len(xWords) == 0:
        print 'column_split return empty'
        return []
    wordRects = []  # 最终的字符的(x,y,w,h)
    for i in range(0, len(xWords)):
        xPos = xWords[i]
        # cv2.rectangle(img, (xPos[0], 0), (xPos[1], height), (0, 0, 255), 2) # 用矩形显示列分割
        # 对每个字符进行水平分割
        wordRect = closed[0:height, xPos[0]:xPos[1]]
        yPos = line_split(wordRect, show_window=None)
        if yPos is None:
            continue
        x1 = xPos[0] + x
        y1 = yPos[0] + y
        w1 = xPos[1] - xPos[0]
        h1 = yPos[1] - yPos[0]
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)  # 用矩形显示最终字符
        wordRects.append((x1, y1, w1, h1))
    if show:
        cv2.imshow('words', img)
    return wordRects # 返回每个字符的(x,y,w,h)


if __name__ == '__main__':
    img = cv2.imread('./area/16.png')
    angle, img = utils.correct_skew(img, is_gray=False)
    cv2.imshow('skew', img)
    oriHeight, oriWidth = img.shape[:2]
    resizedHeight = int(oriHeight / (oriWidth / float(800)))
    img = cv2.resize(img, (800, resizedHeight)) # 将图片宽度固定为800
    img_to_words(img, show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()