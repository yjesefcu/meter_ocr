# -*- coding:utf-8 -*-
import support_library
import cv2
import numpy as np
import wordSplit
import utils

#TRAINING---------------------------------------------------------------------------------------------------------------
samples = np.loadtxt('./data/general_samples.data', np.float32)
responses = np.loadtxt('./data/general_responses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

if __name__ == '__main__':
    img = cv2.imread('./area/44.png')
    angle, img = utils.correct_skew(img)
    oriHeight, oriWidth = img.shape[:2]
    resizedHeight = int(oriHeight / (oriWidth / float(800)))
    img = cv2.resize(img, (800, resizedHeight)) # 将图片宽度固定为800
    wordRects = wordSplit.img_to_words(img) # 字符分割

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    utils.color_reverse(gray)
    gris = cv2.GaussianBlur(gray, (3, 3), 0) # 高斯滤波

    chars = []
    for (x, y, w, h) in wordRects:
        roi = gris[y:y + h, x:x + w]
        roi = support_library.recon_borde(roi)
        roi_small = cv2.resize(roi, (10, 10))
        roi_small = roi_small.reshape((1, 100))
        roi_small = np.float32(roi_small)
        retval, results, neigh_resp, dists = model.findNearest(roi_small, k=1)
        responseNmber = int((results[0][0]))
        if responseNmber > 9:
            print u'识别到刻度'
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, str(responseNmber), (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("MEDIDOR ELECTRICO", img)
        chars.append(str(responseNmber))
    print 'El numero facturado es:', ''.join(chars)
    cv2.waitKey(0)
