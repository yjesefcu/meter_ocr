import cv2

img = cv2.imread('./area/44.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edge_output = cv2.Canny(gray, 50, 150)
cv2.imshow("Canny Edge", edge_output)
cv2.waitKey(0)
