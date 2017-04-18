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



class Pedestrian(object):
    #Pedestrian class
    #__init__ constructor
    def __init__(self, identity, xA, yA, xB, yB):
        self.identity = identity
        self.xcenter = xA + ((xB-xA)/2)
        self.ycenter = yA + ((yB-yA)/2)
        self.xthreshold = (xB-xA)/5
        self.ythreshold = (yB-yA)/5
    #variables needed to roughly estimate location of pedestrian in next frame
    xpast = 0
    ypast = 0
    xexpected = 0
    yexpected = 0
    present = True
    occluded = False
    occlusionframes = 0
    #---most likely a false positive if it keeps dropping.
    drops = 0
    #update (call after find)
    def update(self):
        if present == True:
            self.xexpected = self.xcenter + (self.xcenter - self.xpast)
            self.yexpected = self.ycenter + (self.ycenter - self.ypast)
            self.xpast = self.xcenter
            self.ypast = self.ycenter
            if self.occluded == True:
                self.xcenter = self.xexpected
                self.ycenter = self.yexpected
        else:
            return
    def fix(
    def find(self,ndarray):
        if present == True:
            for (xA, yA, xB, yB) in ndarray:
                if (abs(xA+((xB-xA)/2) - self.xcenter) <= self.xthreshold) and (abs(yA+((yB-yA)/2) - self.ycenter) <= self.ythreshold):
                    self.xcenter = xA+((xB-xA)/2)
                    self.ycenter = yA+((yB-yA)/2)
                    self.xthreshold = (xB-xA)/5
                    self.ythreshold = (yB-yA)/5
                    self.occluded = False
               else:
                    self.occluded = True
    def match():
        "nothing"






#---capture video from video source --video is video path
cap = cv2.VideoCapture('test0.mp4')
#---initialize counter for total number of frames. also used to for current frame count
totalf = 0
#---initialize integer to hold number of pedestrians detected in previous frame and current
pedprev = 0
pednow = 0
#---unique pedestrian detector (should correlate with size of Pedestrian array size
who = 0
#---initialize histogram of oriented gradients
hog = cv2.HOGDescriptor()
#---initialize people detecting svm
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#---font for text
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    (grabbed, image) = cap.read()
    #---check if there is still video
    if not grabbed:
        break
    #---iterate number of frames captured
    totalf += 1
    #---resize image for efficiency
    #image = imutils.resize(image, width=min(200, image.shape[1]))
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    #---maintain overlapping boxes using large overlap threshold(1) or don't allow overlapping with (0)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    rects = non_max_suppression(rects, probs=None, overlapThresh=.65)
    #--- current number of pedestriands detected = size of thresholded rectangles array
    pednow = len(rects)
    #--- draw bounding boxes
    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.putText(image, str(1), (xA, yA), font, 1, (255,255,255), 2)
    #---Show Frame
    cv2.imshow("Frame", image)
    #---After Frame shown, can save some varialbes required for next frame
    pedprev = pednow
    for index in rects:
        print("rectangle: %s\n firstindex: %s\n " %(rects, rects[0,0]))
    #--- press 'Esc' to exit program early
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#--- Print results    
print ("total frames: %i \ndet0: %i \ndet1: %i \ndet2: %i \ndet3: %i \ndet4: %i" %(totalf, p0,p1,p2,p3,p4))
#--- release video capture device from use
cap.release()
cv2.destroyAllWindows()
