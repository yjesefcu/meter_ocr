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

if __name__ == '__main__':
    show = False
    root = './area'
    dirList = os.listdir(root)
    for imgName in dirList:
        # 解决文件夹中有 .DS_STORE的情况
        path = os.path.join(root, imgName)
        if imgName.startswith('.') or os.path.isdir(path):
            continue
        img = cv2.imread('{}/{}'.format(root, imgName))
        angle, img = utils.correct_skew(img, is_gray=False)
        oriHeight, oriWidth = img.shape[:2]
        resizedHeight = int(oriHeight / (oriWidth / float(800)))
        img = cv2.resize(img, (800, resizedHeight)) # 将图片宽度固定为800
        wordRects = wordSplit.img_to_words(img, show) # 字符分割

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
        utils.color_reverse(gray)
        gris = cv2.GaussianBlur(gray, (3, 3), 0) # 高斯滤波
        chars = []
        for (x, y, w, h) in wordRects:
            roi = gray[y:y + h, x:x + w]
            '''下面这段加不加对结果影响不大
            # try:
            #     roi = cv2.resize(roi, (50, 50))
            # except Exception, e:
            #     print e.message
            # T = mahotas.thresholding.otsu(roi)
            # for k in range(1, 50, 1):
            #     for z in range(1, 50, 1):
            #         color = roi[k, z]
            #         if color > T:
            #             roi[k, z] = 0
            #         else:
            #             roi[k, z] = 255
            # roi = utils.color_reverse(roi)
            '''
            roi = support_library.recon_borde(roi)
            roi_small = cv2.resize(roi, (10, 10))
            roi_small = roi_small.reshape((1, 100))
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
        cv2.imwrite('./result-resize/{}-{}{}'.format(name, ''.join(chars), affix), img)
    if show:
        cv2.waitKey(0)
