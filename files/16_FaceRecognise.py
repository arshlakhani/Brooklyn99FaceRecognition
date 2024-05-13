import cv2 as cv
import numpy as np
import os

people = ['Amy_Santiago','Charles_Boyle','Jake_Peralta']
DIR = r'C:\Users\Arsh\PycharmProjects\practice\files'
features = []
label = []
haar_cascade = cv.CascadeClassifier('haar_facedetect.xml')
def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        Label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_detected = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x,y,w,h) in faces_detected:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                label.append(Label)


create_train()
print(f'Length of features = {len(features)}')
print(f'Length of labels = {len(label)}')

features = np.array(features, dtype='object')
label = np.array(label)

face_recogniser = cv.face.LBPHFaceRecognizer_create()
# traing the recogniser
face_recogniser.train(features, label)
# features and labels
face_recogniser.save('face_trainedb99.yml')
np.save('features.npy',features)
np.save('labels.npy',label)

