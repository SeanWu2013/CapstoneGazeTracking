#merge bgv1 and bgv2 (background and blob version 1&2) and people class detector in an attempt to optimie the algorithm
# import the necessary packages
from __future__ import print_function
from imutils.video import count_frames
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
#not really necessary for now
import time
import datetime

#-------------------------------------------------
#---Pedestrian Class
class Pedestrian(object):
    #Pedestrian class
    #__init__ constructor
    def __init__(self, identity, xA, yA, xB, yB):
        self.identity = str(identity)
        self.xnow = xA + ((xB-xA)/2)
        self.ynow = yA + ((yB-yA)/2)
        self.xpast = self.xnow
        self.ypast = self.ynow
        self.xexpected = self.xnow
        self.yexpected = self.ynow
        self.xvelocity = 0
        self.yvelocity = 0
        self.present = True
        self.occluded = False
        #use a threshold based of bounding box size. try implementing one later utilizing pedestrian velocity
        self.xthreshold = (xB-xA)/3
        self.ythreshold = (yB-yA)/3
        #may need to count frames occluded
        self.occlusionframes = 0
        #---most likely a false positive if it keeps dropping.
        self.drops = 0
        self.framespresent = 1
    #---update (call after find)
    def update(self):
        if self.present == True:
            self.framespresent += self.framespresent
            self.xvelocity = (((self.xvelocity)*9)+(self.xnow - self.xpast))/10
            self.yvelocity = (((self.yvelocity)*9)+(self.ynow - self.ypast))/10
            self.xexpected = (self.xnow) + self.xvelocity
            self.yexpected = (self.ynow) + self.yvelocity
            #self.xexpected = self.xnow + (self.xnow - self.xpast)
            #self.yexpected = self.ynow + (self.ynow - self.ypast)
            self.xpast = self.xnow
            self.ypast = self.ynow
            self.framespresent = self.framespresent + 1
            #note if occluded, now should have been calculated in past frame
            #---prep now for next frame in case if occluded
            if self.occluded == True:
                self.xnow = self.xexpected
                self.ynow = self.yexpected
                #---increase threshold size(linearly) to account for turning and slowing down. ended up blowing everything up
                
                self.xthreshold = abs(self.xthreshold + self.xvelocity/4)
                
                self.ythreshold = abs(self.ythreshold + self.yvelocity/4)
                #note now should be within threshold of expected (linear estimation based off previous frame (no averaging))
        else:
            self.occlusionframes = self.occlusionframes + 1
    #---try finding unique pedestrian in array of detected people. ndarray should be 4d. output matched ndarray index into new array
    def find(self,ndarray,out):
        if self.present == True:
            #---for (top left corner, bottom right corner). will iterate through whol ndarray
            for i in range(len(ndarray)):
                xA = ndarray[i,0]
                yA = ndarray[i,1]
                xB = ndarray[i,2]
                yB = ndarray[i,3]
                #---if there exists a detected person in ndarray that matches the extrapolated position of pedestrian, update actual postion and threshold size
                if (abs(xA+((xB-xA)/2) - self.xexpected) <= self.xthreshold) and (abs(yA+((yB-yA)/2) - self.yexpected) <= self.ythreshold):
                    self.xnow = xA+((xB-xA)/2)
                    self.ynow = yA+((yB-yA)/2)
                    self.xthreshold = ((self.xthreshold)+(xB-xA)/3)/2
                    self.ythreshold = ((self.ythreshold)+(yB-yA)/3)/2
                    #---pedestrian no longer occluded as of this frame
                    self.occluded = False
                    self.occlusionframes = 0
                    #update the expected location for next frame
                    out[i] = 1
                #else the person is probably occluded
                else:
                    self.occluded = True
                    self.occlusionframes = self.occlusionframes + 1
    #remove pedestrian from being present in frame. Ideally should only occur when the extrapolated position exceeds frame boundaries(edge of persons bounding box hits edge of frame)
    #should be called every frame
    def checkleave(self,frame):
        #return if person not present. personally i want this false statement instead of the execute if true for some reason
        if self.present == False:
            return
        #set self.present to false once person leaves frame
        else:
            if ((self.xexpected+(self.xthreshold)) > frame.shape[1]) or ((self.xexpected-(self.xthreshold)) < 0) or ((self.yexpected+(self.ythreshold)) > frame.shape[0]) or ((self.yexpected+(self.ythreshold)) <0):
                self.present = False
            #---if occluded for too long, remove from being present. most likely false positive.
            #FOR SOME REASON THIS IS NOT WORKING
            if self.occlusionframes > 15:
                self.present == False
                self.occluded == True
            #worry about deleting people if they get occluded for too long later
                
#---Blob Class
class Blob(object):
    #Blob class
    #__init__ constructor
    def __init__(self, identity, xA, yA, xB, yB):
        #IDENTITY SHOULD INDEX A UNIQUE PEDESTRIAN IDEALLY
        self.identity = str(identity)
        self.xnow = xA + ((xB-xA)/2)
        self.ynow = yA + ((yB-yA)/2)
        self.xpast = self.xnow
        self.ypast = self.ynow
        self.xexpected = self.xnow
        self.yexpected = self.ynow
        self.xvelocity = 0
        self.yvelocity = 0
        self.present = True
        self.occluded = False
        #false positive until proven true
        self.needstest = True
        self.ispedestrian = False
        self.checkframesrequired = 1 #GO WITH A LOW NUMBER FOR NOW FOR PROOF OF CONCEPT
        #use a threshold based of bounding box size. ASPECT RATIO NEEDS TO BE LOCKED LATER
        self.xthreshold = (xB-xA)/3
        self.ythreshold = (yB-yA)/3
        #may need to count frames occluded
        self.occlusionframes = 0
        #---most likely a false positive if it keeps dropping.
        self.drops = 0
        self.framespresent = 1
    #---update (call after find)
    def update(self):
        if self.present == True:
            self.framespresent += self.framespresent
            self.xvelocity = (((self.xvelocity)*9)+(self.xnow - self.xpast))/10
            self.yvelocity = (((self.yvelocity)*9)+(self.ynow - self.ypast))/10
            self.xexpected = (self.xnow) + self.xvelocity
            self.yexpected = (self.ynow) + self.yvelocity
            #self.xexpected = self.xnow + (self.xnow - self.xpast)
            #self.yexpected = self.ynow + (self.ynow - self.ypast)
            self.xpast = self.xnow
            self.ypast = self.ynow
            self.framespresent = self.framespresent + 1
            #note if occluded, now should have been calculated in past frame
            #---prep now for next frame in case if occluded
            if self.occluded == True:
                self.xnow = self.xexpected
                self.ynow = self.yexpected
                #---increase threshold sizeto account for turning and slowing down
                self.xthreshold = abs(self.xthreshold + self.xvelocity/4)
                
                self.ythreshold = abs(self.ythreshold + self.yvelocity/4)
                #note now should be within threshold of expected (linear estimation based off previous frame (no averaging))
        else:
            self.occlusionframes = self.occlusionframes + 1
    #---try finding unique pedestrian in array of detected people. ndarray should be 4d. output matched ndarray index into new array
    def find(self,ndarray,out):
        if self.present == True:
            #---for (top left corner, bottom right corner). will iterate through whole ndarray
            for i in range(len(ndarray)):
                xA = ndarray[i,0]
                yA = ndarray[i,1]
                xB = ndarray[i,2]
                yB = ndarray[i,3]
                #---if there exists a detected person in ndarray that matches the extrapolated position of pedestrian, update actual postion and threshold size
                if (abs(xA+((xB-xA)/2) - self.xexpected) <= self.xthreshold) and (abs(yA+((yB-yA)/2) - self.yexpected) <= self.ythreshold):
                    self.xnow = xA+((xB-xA)/2)
                    self.ynow = yA+((yB-yA)/2)
                    self.xthreshold = ((self.xthreshold)+(xB-xA)/3)/2
                    self.ythreshold = ((self.ythreshold)+(yB-yA)/3)/2
                    #---pedestrian no longer occluded as of this frame
                    self.occluded = False
                    self.occlusionframes = 0
                    #update the expected location for next frame
                    self.update()
                    out[i] = 1
                #else the person is probably occluded
                else:
                    self.occluded = True
                    self.occlusionframes = self.occlusionframes + 1
                    self.update()
    #remove pedestrian from being present in frame. Ideally should only occur when the extrapolated position exceeds frame boundaries(edge of persons bounding box hits edge of frame)
    #should be called every frame
    def checkleave(self,frame):
        #return if person not present. personally i want this false statement instead of the execute if true for some reason
        if self.present == False:
            return
        #set self.present to false once person leaves frame
        else:
            if ((self.xexpected+(self.xthreshold)) > frame.shape[1]) or ((self.xexpected-(self.xthreshold)) < 0) or ((self.yexpected+(self.ythreshold)) > frame.shape[0]) or ((self.yexpected+(self.ythreshold)) <0):
                self.present = False
            #---if occluded for too long, remove from being present. most likely false positive.
            #FOR SOME REASON THIS IS NOT WORKING
            elif self.occlusionframes > 15:
                self.present = False
                self.occluded = True
            #worry about deleting people if they get occluded for too long later

#-------------------------------------------------
#---Initialize Pedestrian Variables
#--array to hold unique pedestrians. initialize with a false pedestrian to reduce iteration times later (explain later? i forgot why)
upedList=[]
upedList.append(Pedestrian(0,0,0,0,0))
upedList[0].identity = "0" #shouldve been done automatically but w.e.
#--number of bounding boxes in frame?
what = 0
#--number of unique pedestrians detected ()
who = 0
#---initialize histogram of oriented gradients
hog = cv2.HOGDescriptor()
#---initialize people detecting svm
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#-------------------------------------------------
#---Initialize Blob Variables
#--set minimum size for blob to be considered blob. default was 500
minblobsize = 500
#--initialize foreground background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
#--background image for subtraction. Will be set by bgv1 algorithm fgbg
bgimg = None
#--list of blobs. will be an ndarray
blobs = []
#--list of unique blobs (try getting destroyeed asap) to attatch a ped to and help remove false positives
ublobsList = []
ublobsList.append(Blob(0,0,0,0,0))
ublobsList[0].identity = "0" #shouldve been done automatically but w.e.

#-------------------------------------------------
#---Initialize Camera
#---capture video from video source. what is in () is video path. 0 is embedded webcam, 1 for usb in my case
cap = cv2.VideoCapture('last0.mp4')
#---set brightness to reduce autowhite balance. Short training time in beginning should fix that issue a bit too
cap.set(15,-5)
#---font for text
font = cv2.FONT_HERSHEY_SIMPLEX
#--counter for total number of frames. also used for current frame count
time = 0
#--frame min width
mwidth = 400

#-------------------------------------------------
#---Initialize Function
#average out background image for 1-2 seconds
while time <= 60:
    #grab the current frame
    ret, frame = cap.read()
    if not ret:
        break
    # resize the frame
    frame = imutils.resize(frame, width=min(mwidth, frame.shape[1]))
    #apparently (input, output, learning speed)
    fgmask = fgbg.apply(frame, 0, 0.1)
    bgimg = fgbg.getBackgroundImage(fgmask)
    #iterate timer
    time = time + 1
    #terminate early by pressing 'Esc'
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
#convert background to grayscale, and blur it
bgimg = cv2.cvtColor(bgimg, cv2.COLOR_BGR2GRAY)
bgimg = cv2.GaussianBlur(bgimg, (21, 21), 0)

while time > 60:
    #grab the current frame
    ret, frame = cap.read()
    if not ret:
        break
    #resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=min(mwidth, frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    #--Main Function
    #initialize another blob container
    blob = []
    # compute the absolute difference between the current frame and background image
    frameDelta = cv2.absdiff(bgimg, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours on thresholded image(iterations originally = 2)
    thresh = cv2.dilate(thresh, None, iterations=5)
    (im2,cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over the contours and place ones that pass into array
    for c in cnts:
	# if the contour is too small, ignore it
        if cv2.contourArea(c) > minblobsize:
        #    continue
        # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            blob.append([x, y, w, h])
    #ERROR THAT OFTEN OCCURS REQUIRING TABIFYING OF REGION IN CODE FOR SOME REASON. STUPID
    blobs = np.array([[c[0], c[1], (c[0] + c[2]), (c[1] + c[3])] for c in blob])
    #---keep overlapping boxes using large overlap threshold(1) or don't allow overlapping with (0)
    blobs = non_max_suppression(blobs, probs=None, overlapThresh=.3)
    #PRIMARY PART OF ALGORITHM> WILL NEED ALOT OF THINKING TO IMPLEMENT HERE. STARTING OFF EXACTLY AS FIRST PED ALGO
    #array to hold 1/0 indicating if blob is actually a pedestrian. refreshed each frame
    testing = [0]*len(blobs)
    #--- check for unique blobs in frame
    for i in range(len(ublobsList)):
        #only check for blobs that may still be present. currently program is expected to slow down as unique blobs increase.
        ublobsList[i].find(blobs,testing)
        ublobsList[i].update()        
    #check for unique blobs in frame
    for i in range(len(testing)):
        if testing[i] == 0:
            #should only make new blob if the blob is actual pedestrian
            #test iftrue. may need to make another array for those
            what = what + 1
            ublobsList.append(Blob(what, blobs[i,0],blobs[i,1],blobs[i,2],blobs[i,3]))
    #-------------make a separate container for present unique pedestrians later? NOTED
    #print extrapolation rectangle
    for i in range(len(ublobsList)):
        if ublobsList[i].present == True:
            ublobsList[i].checkleave(frame)
            cv2.putText(frame, ublobsList[i].identity, (int(ublobsList[i].xnow), int(ublobsList[i].ynow)), font, 1, (255,255,255),1)
            #show threshold box
            cv2.rectangle(frame, (int(ublobsList[i].xexpected - ublobsList[i].xthreshold),int(ublobsList[i].yexpected - ublobsList[i].ythreshold)), (int(ublobsList[i].xexpected + ublobsList[i].xthreshold),int(ublobsList[i].yexpected + ublobsList[i].ythreshold)), (255, 0, 0), 2) 
    
    #---display min-suppressed bounding boxes (no small boxes should appear in any larger ones)
    for (xA, yA, xB, yB) in blobs:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    
    # draw the timestamp on the frame
    #cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
    #show the frame with bounding boxes
    cv2.imshow("frame", frame)
    #show threshold box
    cv2.imshow("Thresh", thresh)
    #show difference between background and foreground
    cv2.imshow("Frame Delta", frameDelta)







    #iterate timer
    time = time + 1
    #terminate early by pressing 'Esc'
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
   
cap.release()
cv2.destroyAllWindows()
#-------------------------------------------------
#MOG background sub
#http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=backgroundsubtractormog#backgroundsubtractormog-backgroundsubtractormog
#cameraparams
#http://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
