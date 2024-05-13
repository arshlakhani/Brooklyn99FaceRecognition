import cv2 as cv 
import numpy as np



img = cv.imread('download.jpeg')
blank = np.zeros(img.shape,dtype='uint8')

cv.imshow('cats',img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
canny = cv.Canny(img,125,175)
cv.imshow('canny',canny)

blur = cv.GaussianBlur(gray, (1,1),cv.BORDER_DEFAULT)
ret, tresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
contours, heirarchy = cv.findContours(canny, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
# RETR_LIST returns all contours , RETE_EXTERNAL returns all external contours and RETR_TREE return hierarichal contours
# contour approximation methods -> CHAIN_APPROX_NONE -> does nothing shows all
# CHAIN_ACROSS_SIMPLE -> returns simple that return them makes sense/ compesses
print(len(contours))
cv.imshow('thresh',tresh)
cv.drawContours(blank, contours, -1,(255,255,255),1)
cv.imshow('Contours Drawn',blank)
cv.waitKey(0)