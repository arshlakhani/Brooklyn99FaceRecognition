import cv2 as cv
import numpy as np

img = cv.imread('IMG-20230510-WA0030.jpg')
img = cv.resize(img,(400,230),interpolation=cv.INTER_AREA)
cv.imshow('img',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
cv.imshow('gray',gray)

haarcascade = cv.CascadeClassifier('haar_facedetect.xml')

faces_detected= haarcascade.detectMultiScale(gray , scaleFactor=1.1,minNeighbors=3)
print("Number of faces detected: ", len(faces_detected))

for (x,y,w,h) in faces_detected:
    cv.rectangle(img,(x,y),(x+w,y+h), (2,255,0),thickness=2)
    
cv.imshow('img_detected',img)


cv.waitKey(0)