# backroundreducer
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(15,-5)#set fixed brightness to prevent some autowhite balance

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
fgbg = cv2.createBackgroundSubtractorMOG2()
time = 0

while time <= 60:
    
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame, 0, 0.1)
    bgimg = fgbg.getBackgroundImage(fgmask)
    cv2.imshow('original',frame)
    cv2.imshow('background',bgimg)
    #fgmask[150:250,150:250] = [255]
    cv2.imshow('foreground',fgmask)

    #pixel = fgmask[300,300]
    #print(pixel)
    time = time + 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
while time > 60:
    
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame, 0, 0.0)
    bgimg = fgbg.getBackgroundImage(fgmask)
    cv2.imshow('original',frame)
    cv2.imshow('background',bgimg)
    #fgmask[150:250,150:250] = [255]
    cv2.imshow('foreground',fgmask)

    #pixel = fgmask[300,300]
    #print(pixel)
    time += time
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
#background sub
#http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=backgroundsubtractormog#backgroundsubtractormog-backgroundsubtractormog
#cameraparams
#http://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
