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
        self.xnow = xA + ((xB-xA)/2)
        self.ynow = yA + ((yB-yA)/2)
        #use a threshold based of bounding box size. try implementing one later utilizing pedestrian velocity
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
    #---update (call after find)
    def update(self):
        if self.present == True:
            self.xexpected = self.xnow + (self.xnow - self.xpast)
            self.yexpected = self.ynow + (self.ynow - self.ypast)
            self.xpast = self.xnow
            self.ypast = self.ynow
            #note if occluded, now should have been calculated in past frame
            #---prep now for next frame in case if occluded
            if self.occluded == True:
                self.xnow = self.xexpected
                self.ynow = self.yexpected
                #---increase threshold size to account for turning and slowing down
                self.xthreshold = self.xthreshold + 2
                self.ythreshold = self.ythreshold + 2
                #note now should be within threshold of expected (linear estimation based off previous frame (no averaging))
        else:
            return
    #---try finding unique pedestrian in array of detected people 
    def find(self,ndarray):
        if self.present == True:
            #---for (top left corner, bottom right corner)
            for (xA, yA, xB, yB) in ndarray:
                #---if there exists a detected person in ndarray that matches the extrapolated position of pedestrian, update actual postion and threshold size
                if (abs(xA+((xB-xA)/2) - self.xexpected) <= self.xthreshold) and (abs(yA+((yB-yA)/2) - self.yexpected) <= self.ythreshold):
                    self.xnow = xA+((xB-xA)/2)
                    self.ynow = yA+((yB-yA)/2)
                    self.xthreshold = (xB-xA)/5
                    self.ythreshold = (yB-yA)/5
                    #---pedestrian no longer occluded as of this frame
                    self.occluded = False
                    #update the expected location for next frame
                    self.update()
                #else the person is probably occluded
                else:
                    self.occluded = True
                    self.update()
    #remove pedestrian from being present in frame. Ideally should only occur when the extrapolated position exceeds frame boundaries(edge of persons bounding box hits edge of frame)
    #should be called every frame
    def checkleave(self,frame):
        #return if person not present. personally i want this false statement instead of the execute if true for some reason
        if self.present == False:
            return
        #set self.present to false once person leaves frame
        else:
            if ((self.xexpected+(self.xthreshold*5)) > frame.shape[1]) or ((self.xexpected-(self.xthreshold*5)) < 0) or ((self.yexpected+(self.ythreshold*5)) > frame.shape[0]) or ((self.yexpected+(self.ythreshold*5)) <0):
                self.present = False
            #worry about deleting people if they get occluded for too long later
 





#---capture video from video source --video is video path
cap = cv2.VideoCapture('test0.mp4')
#---initialize counter for total number of frames. also used to for current frame count
totalf = 0
#---initialize integer to hold number of pedestrians detected in previous frame and current
pedprev = 0
pednow = 0
#---unique pedestrian detector (should correlate with size of Pedestrian array size
who = 0
newped = False
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
    #---index for iterating through rects
    for i in range(len(rects)):
       # seems to output top left corner and bottom right corner of box
       print("rectangle: %s\n firstindex: %s " %(rects[i], rects[i,0]))
       i += i
    #--- press 'Esc' to exit program early
    print("\n")
    #---After Frame shown, can save some varialbes required for next frame (prepare for next frame)
    pedprev = pednow
    newped = False
    #--- press 'Esc' to exit program early
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#--- Print results    
print ("total frames: %i \ndet0: %i \ndet1: %i \ndet2: %i \ndet3: %i \ndet4: %i" %(totalf, p0,p1,p2,p3,p4))
#--- release video capture device from use
cap.release()
cv2.destroyAllWindows()
