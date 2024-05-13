import numpy as np
import cv2 as cv

# read images
img = cv.imread('download.jpeg')
cv.imshow('Cat',img)
cv.waitKey(10)
# waits for a specific delay to display
# 0 means infinite

# capturing Video
capture = cv.VideoCapture(0)
# webcam integer
# or can write filepath
while True: 
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)
    # specific fram
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()