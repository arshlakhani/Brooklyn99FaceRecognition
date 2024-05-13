import cv2 as cv
import numpy as np
# webcam video in opencv?
img = cv.VideoCapture(0)

haarcascade = cv.CascadeClassifier('haar_facedetect.xml')
while True:
    ret, frame = img.read()
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces_detected= haarcascade.detectMultiScale(gray_frame , scaleFactor=1.1,minNeighbors=5)
    # print("Number of faces detected: ", len(faces_detected))  

    for (x,y,w,h) in faces_detected:
        cv.rectangle(frame,(x,y),(x+w,y+h), (25,255,0),thickness=2)
    cv.imshow('frames',frame)
    fps = img.get(cv.CAP_PROP_FPS)
    print(f"{fps} frames per second")
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


img.release()
cv.destroyAllWindows()