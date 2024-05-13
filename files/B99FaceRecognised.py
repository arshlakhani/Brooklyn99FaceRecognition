import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_facedetect.xml')

people = ['Amy_Santiago','Charles_Boyle','Jake_Peralta']
#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recog = cv.face.LBPHFaceRecognizer_create()
face_recog.read('face_trainedb99.yml')

img = cv.imread(r'main-qimg-c1e74c97129e8db6866bda4a982fd2ef-lq.jpeg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)

faces_recognised = haar_cascade.detectMultiScale(gray, 1.5,7)
# how to set confidence limit in opencv?
for (x,y,w,h) in faces_recognised:
    
    roi_face = gray[y:y+h, x:x+w]
    label,confidence = face_recog.predict(roi_face)
    
    print("label= ",people[label],",confidence=",confidence)
    cv.putText(img,str(people[label]),(x-20,y-20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=1)
    cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)

cv.imshow('Face Recogniser',img)
cv.waitKey(0)