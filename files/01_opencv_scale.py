import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    # 1 is width
    height = int(frame.shape[0] * scale)
    # 0 is height
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRES(width,height):
    capture.set(3,width)
    capture.set(4,height)
    # for live video only
    
capture = cv.VideoCapture(0)

while True: 
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame,scale=1.2)
    #frame_90 = rescaleFrame(frame,scale=0.90)
    cv.imshow('Video',frame)
    cv.imshow('75',frame_resized)
    #cv.imshow('90',frame_90)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
# can even do this with images

capture.release()
cv.destroyAllWindows()