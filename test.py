import cv2
import numpy as np
import utils

img = cv2.imread('./test0309/36.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', gray)
edge_output = cv2.Canny(gray, 50, 150)
cv2.imshow("Canny Edge", edge_output)
blur = cv2.blur(gray, (3, 3))
cv2.imshow('blur', blur)
gauss = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow('guass', gauss)

kernel = np.ones((5, 5), np.float32) / 25
# gray = cv2.filter2D(gray, -1, kernel)
gray = cv2.medianBlur(gray, 5)
threshold = utils.custom_threshold(gray)
cv2.imshow('threshold', threshold)
cv2.imshow('filter2D', gray)
cv2.waitKey(0)
