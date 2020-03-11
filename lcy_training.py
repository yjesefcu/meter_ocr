# -*- coding:utf-8 -*-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-#
# PROJECT   : DETECTION OF NUMBERS IN ELECTRIC METER(TRAINING)                                                         #
# VERSION   : 1.0                                                                                                      #
# AUTHOR    : Valeria Quinde Granda             valeestefa15@gmail.com                                                 #
# PROFESSOR : Rodrigo Barba                     lrbarba@utpl.edu.ec                                                    #
# COMPANY   : Sic ElectriTelecom  Loja-Ecuador                                                                         #
# DATE      : 26/08/2015                                                                                               #
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-#
import numpy as np
import cv2
import mahotas
import os

# OPEN TRAINING IMAGE FOR PROCESSING------------------------------------------------------------------------------------
width = 30
height = 60
samples = np.empty((0, width * height))
responses = []
sampleDir = './samples2'
dirList = os.listdir(sampleDir)
for dir in dirList:
    # 解决文件夹中有 .DS_STORE的情况
    if dir.startswith('.') or dir == '10':
        continue
    print 'dir: {}'.format(dir)
    path = os.path.join(sampleDir, dir)
    if os.path.isdir(path):
        fileList = os.listdir(path)
        for f in fileList:
            if f.startswith('.'):
                continue
            image = cv2.imread('{}/{}'.format(path, f), 0)
            try:
                image = cv2.resize(image, (width, height))
            except Exception, e:
                print e.message
            # DETECTION THRESHOLD----------------------------------------------------------------------------------------------------
            T = mahotas.thresholding.otsu(image)
            for k in range(1, height, 1):
                for z in range(1, width, 1):
                    color = image[k, z]
                    if (color > T):
                        image[k, z] = 255
                    else:
                        image[k, z] = 0
            thresh2 = image.copy()
            cv2.imshow('thresh2', thresh2)
            cv2.destroyWindow('norm')
            cv2.imshow('Numero', image)
            sample = thresh2.reshape((1, width * height))
            samples = np.append(samples, sample, 0)
            responses.append(int(dir))
print "training complete"
np.savetxt('data/general_samples.data', samples)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
np.savetxt('data/general_responses.data', responses)

cv2.waitKey(0)

