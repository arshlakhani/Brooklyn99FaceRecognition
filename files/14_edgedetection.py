import cv2 as cv 
import numpy as np

img = cv.imread('Taj-Mahal-Agra-India.jpg')
img = cv.resize(img,(700,400),interpolation=cv.INTER_AREA)

cv.imshow('Taj',img)

# Gradients and edge detection
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# laplacian
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('lap',lap)

# Sobel gradient magintude representation
sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)
cv.imshow('x',sobelx)
cv.imshow('u',sobely)
combined_sobel = cv.bitwise_or(sobelx,sobely)
cv.imshow('combined',combined_sobel)
# Canny Edge Detection
canny = cv.Canny(gray,150,175)
cv.imshow('canny',canny)
cv.waitKey(0)