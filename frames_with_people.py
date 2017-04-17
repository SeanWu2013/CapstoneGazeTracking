# --video video path

#---import the necessary packages
from __future__ import print_function
from imutils.video import count_frames
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

#---capture video from video source --video is video path
cap = cv2.VideoCapture('darkblend.mp4')
#---initialize counter for total number of frames
totalf = 0
#---initialize counter for number of people detected in frames
p0 = 0
p1 = 0
p2 = 0
p3 = 0
p4 = 0
#---initialize histogram of oriented gradients
hog = cv2.HOGDescriptor()
#---initialize people detecting svm
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
    (grabbed, image) = cap.read()
    #---check if there is still video
    if not grabbed:
        break
    #---iterate number of frames captured
    totalf += 1
    #---resize image for efficiency
    image = imutils.resize(image, width=min(200, image.shape[1]))
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    #---maintain overlapping boxes using large overlap threshold(1) or don't allow overlapping with (0)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    rects = non_max_suppression(rects, probs=None, overlapThresh=1)
    #---iterate counters
    if len(rects) == 0:
        p0 += 1
    elif len(rects) == 1:
        p1 += 1
    elif len(rects) == 2:
        p2 += 1
    elif len(rects) == 3:
        p3 += 1
    elif len(rects) >3 :
        p4 += 1
    #---print current frame number for debugging
    #print(totalf)
    #---print counted people in current frame for debugging
    #print(len(rects))
    #--- Print height and width of window
    #print ("height Y: %i" %image.shape[0])
    #print ("width  X: %i" %image.shape[1])
    #--- draw bounding boxes
    for (xA, yA, xB, yB) in rects:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #---Show Frame
    cv2.imshow("0.3 Threshold", image)
    #--- press 'Esc' to exit program early
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#--- Print results    
print ("total frames: %i \ndet0: %i \ndet1: %i \ndet2: %i \ndet3: %i \ndet4: %i" %(totalf, p0,p1,p2,p3,p4))
#--- release video capture device from use
cap.release()
cv2.destroyAllWindows()
