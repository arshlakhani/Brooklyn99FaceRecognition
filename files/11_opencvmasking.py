import cv2 as cv
import numpy as np

img = cv.imread('Taj-Mahal-Agra-India.jpg')
img = cv.resize(img,(700,400),interpolation=cv.INTER_AREA)

cv.imshow('Taj',img)

blank = np.zeros(img.shape[:2],dtype='uint8')
cv.imshow('blank',blank)

mask = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
cv.imshow('mask',mask)
mask = cv.rectangle(mask,(100,100),(400,300),255,-1)
cv.imshow('Weird Mask',mask)
masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('maskedimg',masked)
cv.waitKey(0)
# size of mask always has to be the same size of your image