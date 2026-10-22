import cv2

img = cv2.imread('D:/Python_Projects/Smart_document_scanner/images/02_rotated_15deg.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)

canny = cv2.Canny(blur,50,150)

contours , _ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y),(w+x , y+h),(0,255,0),2)

cv2.imshow('Orginal', img)
cv2.waitKey(0)
cv2.destroyAllWindows()