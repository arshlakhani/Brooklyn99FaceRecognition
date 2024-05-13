import cv2 as cv
import numpy as np

# blank image 
blank = np.zeros((500,500,3),dtype='uint8')
# 3 is the number of color channels
# paint the image a certain color
blank[200:300,300:400] = 255,0,0
blank[100:200,400:450] = 32,255,0

# draw a rectangle
cv.rectangle(blank,(0,0),(250,250), (0,255,0), thickness=2)
# to fill thickness = cv.FILLED or -1

# draw a circle
cv.circle(blank,(250,250),40,(255,0,0),thickness=2)

# draw  aline
cv.line(blank,(0,0),(100,100),(255,255,0),thickness=1)
cv.rectangle(blank,(100,100),(blank.shape[1]//2 , blank.shape[0]//4),(0,0,255),thickness= -1)
# write a text on a image
cv.putText(blank,'Hello World',(225,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(50,100,100),2)
cv.imshow('blank',blank)
cv.waitKey(20000)