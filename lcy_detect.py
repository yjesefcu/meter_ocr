# -*- coding:utf-8 -*-
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-#
# PROJECT   : DETECTION OF NUMBERS IN ELECTRIC METER                                                                   #
# VERSION   : 1.0                                                                                                      #
# AUTHOR    : Valeria Quinde Granda             valeestefa15@gmail.com                                                 #
# PROFESSOR : Rodrigo Barba                     lrbarba@utpl.edu.ec                                                    #
# COMPANY   : Sic ElectriTelecom  Loja-Ecuador                                                                         #
# DATE      : 26/08/2015                                                                                               #
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-#
import support_library
import cv2
import numpy as np
import mahotas
import time
import wordSplit

#TRAINING---------------------------------------------------------------------------------------------------------------
samples = np.loadtxt('./data/general_samples.data', np.float32)
responses = np.loadtxt('./data/general_responses.data', np.float32)
responses = responses.reshape((responses.size,1))
digito=0
# model = cv2.KNearest()
# model.train(samples, responses)
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
rois=100
xf=1
xfx=xf
contador=1
#LOAING IMAGE-----------------------------------------------------------------------------------------------------------
nombre=(raw_input("WRITE THE NAME OF THE PICTURE:"))
image = cv2.imread('images/'+str(nombre))
print "WAIT A MOMENT PLEASE..... PROCESSING"
sp = image.shape
height = float(sp[0])  # height(rows) of image
width = float(sp[1])
image = cv2.resize(image, (800, int(height/(width/800))))
gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mascar=np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mascar, (xf, rois), (xf+800, rois+90), 255, -1)
image2=cv2.bitwise_and(gris,gris,mask=mascar)
T3=mahotas.thresholding.otsu(image2)
gris_copy=gris.copy()
gris_2=gris.copy()
#NEGATIVE IMAGE---------------------------------------------------------------------------------------------------------
for j in range(1, gris.shape[1], 1):
    for i in range(1, gris.shape[0], 1):
        color=gris[i,j]
        gris[i,j]=255-color
gris=cv2.GaussianBlur(gris, (3, 3),0)
T1=mahotas.thresholding.otsu(gris)
clahe = cv2.createCLAHE(clipLimit=1.0)
grises= clahe.apply(gris)
conteo=1
T2 = mahotas.thresholding.otsu(grises)
T=(T2+T1+5)/2
#THRESHOLD--------------------------------------------------------------------------------------------------------------
maxWidth = rois + 90
if maxWidth > gris.shape[0]:
    maxWidth = gris.shape[0]
for k in range(rois, maxWidth, 1):
    for z in range(xf, gris.shape[1], 1):
        color=grises[k,z]
        if color>T:
            grises[k,z]=0
        else:
            grises[k,z]=255
cv2.imshow("gris",grises)
#MASCARA FOR ROI--------------------------------------------------------------------------------------------------------
# mascara=np.zeros(image.shape[:2], dtype="uint8")
# cv2.rectangle(mascara, (xf, rois), (xf+800, rois+90), 255, -1)
# image1=cv2.bitwise_and(grises,grises,mask=mascara)
image1 = image
cv2.imshow("MEDIDOR ELECTRICO",image)
# cv2.waitKey(0)
#FILTER-----------------------------------------------------------------------------------------------------------------
blurred = cv2.GaussianBlur(image1, (7,7),0)
blurred = cv2.medianBlur(blurred,1)
#THRESHOLD--------------------------------------------------------------------------------------------------------------
v = np.mean(blurred)
sigma=0.33
lower = (int(max(0, (1.0 - sigma) * v)))
upper = (int(min(255, (1.0 + sigma) * v)))
#EDGE DETECTION---------------------------------------------------------------------------------------------------------
edged = cv2.Canny(blurred, lower, upper)
cv2.imshow('edged', edged)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
yf=rois
vec=[]
#EDGE RECOGNITION-------------------------------------------------------------------------------------------------------
consumo=0
words = []
for (c,_) in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w > 11 and h > 13 and w<100:
      if(x-xfx)>10:
          xfx = x + w
          yf = y
          roi2 = gris[y:y + h, x:x + w]
          roi = support_library.recon_borde(roi2)
          roi_small = cv2.resize(roi, (10, 10))
          roi_small = roi_small.reshape((1, 100))
          roi_small = np.float32(roi_small)
          retval, results, neigh_resp, dists = model.findNearest(roi_small, k=1)
          string = str(int((results[0][0])))
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
          cv2.putText(image, str(string), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
          cv2.imshow("MEDIDOR ELECTRICO", image)
          words.append(string)
#NUMBER DETECTED--------------------------------------------------------------------------------------------------------
print 'El numero facturado es:', ''.join(words)
cv2.waitKey(0)
