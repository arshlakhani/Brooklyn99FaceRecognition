import cv2 as cv
import numpy as np

img = cv.imread('download.jpeg')
cv.imshow('Cat',img)

blank = np.zeros(img.shape[:2],dtype='uint8')
# convert bgr and splitting into its three channels
b,g,r=cv.split(img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('blue',b)
cv.imshow('green',g)
cv.imshow('red',r)

cv.imshow('blue2',blue)
cv.imshow('green2',green)
cv.imshow('red2',red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('merged',merged)
cv.waitKey(0)