import cv2 as cv
import numpy as np

img = cv.imread('download.jpeg')
cv.imshow('Cat',img)


gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Simple Thresholding

threshold,thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
# if above 150 sets the color to 255
cv.imshow('thresholded',thresh)

threshold,thresh_inv = cv.threshold(gray,120,255,cv.THRESH_BINARY_INV)
cv.imshow('thresh_inv',thresh_inv)

# Adaptive thresholding
adaptiveThreshold= cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,13,9)
# Gaussian /mean --> Adaptive
# mean of neighbourhood pixels is taken into account for thresholding(No given thresholding)
cv.imshow("Adaptive Mean", adaptiveThreshold)

cv.waitKey(0)