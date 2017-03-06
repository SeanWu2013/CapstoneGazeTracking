# backroundreducer
import cv2
import numpy as np

video = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    
    ret, frame = video.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('original',frame)

    #fgmask[150:250,150:250] = [255]
    cv2.imshow('fg',fgmask)

    #pixel = fgmask[300,300]
    #print(pixel)
    k = cv2.waitKey(30) & 0xff
    if k == ord('s'):
        break
    
video.release()
cv2.destroyAllWindows()
