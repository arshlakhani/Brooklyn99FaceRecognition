import cv2 as cv
import numpy as np

img = cv.imread('download.jpeg')
cv.imshow('Cat',img)
# converting image to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Cat2',gray)

# blur an image
blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
#(3,3) is kernel size (always has to be odd)
cv.imshow('Cat3',blur)

# edge detection / Edge Cascade
canny = cv.Canny(blur,125,125)
# pass in blur to show less edges
cv.imshow('Cat4',canny)

# Dilating the imgs
dilated = cv.dilate(canny, (7,7),iterations=3)
cv.imshow('Cat5',dilated)

#eroding
eroded = cv.erode(dilated,(3,3),iterations=2)
cv.imshow('Cat6',eroded)

# resize
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
#INTER_AREA interporation if smaller INTER_LINEAR & INTER_CUBIC if img is getting bigger
cv.imshow('resized',resized)

#cropping (img is array and we can select img portion)
cropped = resized[50:200,200:400]
cv.imshow('cropped',cropped)
cv.waitKey(0)