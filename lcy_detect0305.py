# -*- coding:utf-8 -*-
import support_library
import cv2
import os
import numpy as np
import wordSplit
import utils
import mahotas


#TRAINING---------------------------------------------------------------------------------------------------------------
samples = np.loadtxt('./data/general_samples.data', np.float32)
responses = np.loadtxt('./data/general_responses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

width = 30
height = 60
#THRESHOLD FOR DETECTION------------------------------------------------------------------------------------------------
def recon_borde(image):
    image = cv2.resize(image, (width, height))
    t= mahotas.thresholding.otsu(image)
    for k in range(1, height, 1):
        for z in range(1, width, 1):
            color=image[k,z]
            if color>t:
                image[k,z]=0
            else:
                image[k,z]=255
    thresh = image.copy()
    return thresh


if __name__ == '__main__':
    show = False
    root = './test0311'
    dirList = os.listdir(root)
    for imgName in dirList:
        # 解决文件夹中有 .DS_STORE的情况
        path = os.path.join(root, imgName)
        if imgName.startswith('.') or os.path.isdir(path):
            continue
        print 'image name: {}'.format(imgName)
        img = cv2.imread('{}/{}'.format(root, imgName))
        # 1、倾斜矫正
        angle, img = utils.correct_skew(img, is_gray=False)
        oriHeight, oriWidth = img.shape[:2]
        resizedHeight = int(oriHeight / (oriWidth / float(800)))
        # 2、大小归一化，宽度固定为800
        img = cv2.resize(img, (800, resizedHeight)) # 将图片宽度固定为800
        # 3、字符分割
        wordRects = wordSplit.img_to_words(img, show) # 字符分割
        # 4、图像灰化后颜色反转
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
        utils.color_reverse(gray)
        gris = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
        chars = []
        index = 1
        for (x, y, w, h) in wordRects:
            if w == 0 or h == 0:
                continue
            roi = gray[y:y + h, x:x + w]
            roi = utils.custom_threshold(roi)
            cv2.imshow('roi{}'.format(index), roi)
            roi = cv2.resize(roi, (width, height))
            index += 1
            roi_small = roi.reshape((1, width * height))
            roi_small = np.float32(roi_small)
            retval, results, neigh_resp, dists = model.findNearest(roi_small, k=1)
            responseNmber = int((results[0][0]))
            if responseNmber > 9:
                print u'识别到刻度'
                # continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, str(responseNmber), (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            if show:
                cv2.imshow("{} result".format(imgName), img)
            chars.append(str(responseNmber))
        print 'result:', ''.join(chars)
        name = imgName[0: imgName.index('.')]
        affix = imgName[imgName.index('.'):]
        cv2.imwrite('./test0311-result/{}-{}{}'.format(name, ''.join(chars), affix), img)
    if show:
        cv2.waitKey(0)
    cv2.waitKey(0)
