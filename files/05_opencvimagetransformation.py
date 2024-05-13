import cv2 as cv
import numpy as np

img = cv.imread('Taj-Mahal-Agra-India.jpg')
img = cv.resize(img,(700,400),interpolation=cv.INTER_AREA)

cv.imshow('Taj Mahal',img)

# translation
# shifting across left right etc

def translate(img,x,y):
    transMAT = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMAT, dimensions)
#-x left
#-y up
# x right
# y down
translated = translate(img,100,-100)
cv.imshow('TMShifted',translated)

# rotation 
def rotate(img,angle,rotPoint=None):
    (height,width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

rotatedimg = rotate(img,-45)
# negative for clockwise
cv.imshow('rotated',rotatedimg)

# rotate a rotated img
rotatedimg_ofa_rotatedimg = rotate(rotatedimg,+45)
cv.imshow('Rotating the Rotated',rotatedimg_ofa_rotatedimg)

spimg = img

#resize and image
resized = cv.resize(img,(500,300),interpolation=cv.INTER_AREA)
cv.imshow('sp',resized)

flipped = cv.flip(img,0)
# 1 -> y axis flip
# 0 -> x axis flip
# -1 -> both
cv.imshow('flipped',flipped)
cv.waitKey(0)
