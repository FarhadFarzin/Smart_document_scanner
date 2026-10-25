import cv2
import numpy as np

img = cv2.imread('D:/Python_Projects/Smart_document_scanner/images/02_rotated_15deg.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)

canny = cv2.Canny(blur,50,150)

contours , _ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
if len(approx) == 4:
    
    width = 400
    height = 450
    src = approx.reshape(4,2).astype('float32')
    dst = np.float32([[0,0] , [width - 1 , 0] , [width - 1 , height - 1],[0 , height - 1] ])
    
    
    
cv2.imshow('Final', img)
cv2.waitKey(0)
cv2.destroyAllWindows()