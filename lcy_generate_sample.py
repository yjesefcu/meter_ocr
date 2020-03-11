# -*- coding:utf-8 -*-
import cv2
import wordSplit
import utils
import time
import datetime
import os


# 生成采集字符
if __name__ == '__main__':
    imgName = '40.png'
    root = './test0311'
    dirList = os.listdir(root)
    return_thresh = True # 是否返回二值化字符
    for imgName in dirList:
        # 解决文件夹中有 .DS_STORE的情况
        path = os.path.join(root, imgName)
        if imgName.startswith('.') or os.path.isdir(path):
            continue
        print 'dir: {}'.format(dir)
        img = cv2.imread('{}/{}'.format(root, imgName))
        oriHeight, oriWidth = img.shape[:2]
        resizedHeight = int(oriHeight / (oriWidth / float(800)))
        img = cv2.resize(img, (800, resizedHeight)) # 将图片宽度固定为800
        # 字符分割
        wordRects = wordSplit.img_to_words(img)
        if len(wordRects):
            pass
        # 对图像处理
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
        gris = cv2.GaussianBlur(gray, (3, 3), 0) # 高斯滤波
        index = 0
        input = gris
        # if return_thresh:
        #     thresh = utils.custom_threshold(gris)
        #     input = thresh
        # else:
        #     # utils.color_reverse(gris) # 颜色反转
        #     input = gris
        name = imgName[0 : imgName.index('.')]
        print name
        for (x, y, w, h) in wordRects:
            roi = input[y:y + h, x:x + w]
            if return_thresh:
                roi = utils.custom_threshold(roi)
            # 按16*16压缩
            # c = cv2.resize(roi, (16, 16))
            cv2.imwrite('./test0311-samples/{}-{}.png'.format(name, index), roi)
            index += 1
