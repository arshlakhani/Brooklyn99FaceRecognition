import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Computing histogram
img = cv.imread('download.jpeg')
cv.imshow('Cat',img)
'''
blank = np.zeros(img.shape[:2],dtype='uint8')
crc = cv.circle(blank,[img.shape[1]//2,img.shape[0]//2],100,255,-1)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

mask = cv.bitwise_and(gray,gray,mask=crc)
cv.imshow('Maskedgray',mask)
# gray scale histogram
gray_hist = cv.calcHist([gray], [0],mask,[256],[0,256])
# we usually use multiple images here so were giving it in a list
plt.figure()
plt.title('GrayScale Histogram')
plt.xlabel('Bins')
plt.ylabel('No of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
'''
blank2 = np.zeros(img.shape[:2],dtype='uint8')
mask = cv.circle(blank2,[img.shape[1]//2,img.shape[0]//2],100,255,-1)
masked=cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked',masked)
# COLOR HISTOGRAM
plt.figure()
plt.title('GrayScale Histogram')
plt.xlabel('Bins')
plt.ylabel('No of pixels')
colors = ('b','g','r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
    
    
plt.show()

cv.waitKey(0)
