import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Snapchat-1588857622.jpg')
cv.imshow('img',img)
cv.waitKey(0)