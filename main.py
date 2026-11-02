import cv2
import os
import numpy as np

folder = 'D:/Python_Projects/Smart_document_scanner/images'

images = sorted(os.listdir(folder))

for image_name in images:

    img = cv2.imread(os.path.join(folder , image_name))
          
    denoised = cv2.medianBlur(img, 9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(denoised, 50, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
    
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
        
            pts = approx.reshape(4 , 2)
            s = pts.sum(axis = 1)
            d = np.diff(pts, axis = 1)

            width = 500
            height = 500

            src = np.float32([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)], pts[np.argmax(d)]])
            dst = np.float32([[0, 0],[width-1 ,0],[width-1 ,height-1],[0 ,height-1]])
            matrix = cv2.getPerspectiveTransform(src , dst)
            result = cv2.warpPerspective(img, matrix, (width , height))
            
            if image_name == "11_rotated_90.jpg":
                
                result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    
            
    cv2.imshow('Befor', img)
    cv2.imshow('After', result)
    cv2.imwrite('output/' + image_name , result)
    cv2.waitKey(0)
cv2.destroyAllWindows()