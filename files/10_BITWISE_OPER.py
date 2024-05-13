import cv2 as cv
import numpy as np

# and or exor not
blank = np.zeros((400,400),dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30),(370,370),255,-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)
cv.imshow('rectangle',rectangle)
cv.imshow('circle',circle)
# bitwise AND
bitAND = cv.bitwise_and(rectangle, circle)
cv.imshow('AND',bitAND)

#bitwise OR
bitOR= cv.bitwise_or(rectangle,circle)
cv.imshow('OR',bitOR)

#bitwiseEXOR
bitXOR= cv.bitwise_xor(rectangle,circle)
cv.imshow('XOR',bitXOR)

# bitwise NOT
bitNOT= cv.bitwise_not(rectangle)
cv.imshow('NOT',bitNOT)

cv.waitKey(0)