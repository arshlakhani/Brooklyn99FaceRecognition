import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('download.jpeg')
cv.imshow('Cat',img)

# BGR to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# BGR to HSV
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('hsv',hsv)

# BGR to LAB
lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('lab',lab)

# BGR to RGB
rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('rgb',rgb)
# this may look weird but open cv shows bgr images and other with invert it as rgb
# so we need to convert back to rgb for better visualization of the image
plt.imshow(rgb)
plt.show()


cv.waitKey(0)