import cv2 as cv
import numpy as np

img = cv.imread('download.jpeg')
cv.imshow('Cat',img)

#we used the gaussian blur method

# Averaging method
# averaging all of the 8 surrounding pixels
average = cv.blur(img,(3,3))
cv.imshow('AVG Blur',average)

#Gaussian blur
# more natural than averaging
gaussian_blur= cv.GaussianBlur(img,(3,3),0) 
# sigmaX and Y are standard deviations in X & Y
cv.imshow('gauss',gaussian_blur)

# median blur
med = cv.medianBlur(img,3)
cv.imshow('median',med)

#bilateral blurring
bilat = cv.bilateralFilter(img ,10,35,25) 
# most effective
#retains edges but smooths the noise
cv.imshow('bilateral',bilat)
cv.waitKey(0)