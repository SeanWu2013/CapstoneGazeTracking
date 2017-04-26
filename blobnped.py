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

#------------------------------------------------------------                
#---Blob Class
class Blob(object):
    rewait = 10
    #Blob class
    #__init__ constructor
    def __init__(self, blobnum, xA, yA, xB, yB):
        #IDENTITY SHOULD INDEX A UNIQUE PEDESTRIAN IDEALLY
        self.blobnum = str(blobnum)
        self.identity = "null"
        self.xnow = xA + ((xB-xA)/2)
        self.ynow = yA + ((yB-yA)/2)
        self.width = (xB-xA)
        self.height = (yB-yA)
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
        self.wait = self.rewait
        #frames required to have positive ped. detect before being considered false positive
        self.checkframesrequired = 2 #GO WITH A LOW NUMBER FOR NOW FOR PROOF OF CONCEPT
        #once the blob officially checks off as a pedestrian, will be officially added to list of detected pedestrians (new instance?)
        self.beenadded = False
        #use a threshold based of bounding box size. ASPECT RATIO NEEDS TO BE LOCKED LATER
        self.xthreshold = (xB-xA)/3
        self.ythreshold = (yB-yA)/3
        #may need to count frames occluded
        self.occlusionframes = 0
        #---most likely a false positive if it keeps dropping.
        self.drops = 0
        self.framespresent = 0
    #---update (call after find)
    def update(self):
        if self.present == True:
            self.xvelocity = (((self.xvelocity)*4)+(self.xnow - self.xpast))/5
            self.yvelocity = (((self.yvelocity)*4)+(self.ynow - self.ypast))/5
            self.xexpected = (self.xnow) + self.xvelocity
            self.yexpected = (self.ynow) + self.yvelocity
            #self.xexpected = self.xnow + (self.xnow - self.xpast)
            #self.yexpected = self.ynow + (self.ynow - self.ypast)
            self.xpast = self.xnow
            self.ypast = self.ynow
            self.framespresent = self.framespresent + 1
            #check if blob is a person to remove false positives
            if self.needstest == True:
                "blank"
                #CHECK ALGO
                #SHOULD ALTER NEEDS TEST, IS PEDESTRIAN, CHECK FRAMES REQUIRED
            #note if occluded, now should have been calculated in past frame
            #---prep "now" position for next frame in case if occluded
            if self.occluded == True:
                self.xnow = self.xexpected
                self.ynow = self.yexpected
                #---increase threshold sizeto account for turning and slowing down
                self.xthreshold = abs(self.xthreshold + abs(self.xvelocity/4))
                self.ythreshold = abs(self.ythreshold + abs(self.yvelocity/4))
                self.occlusionframes = self.occlusionframes + 1
                #note "now" should be within threshold of expected (linear estimation based off previous frame (no averaging))
    #---try finding/matching unique blob in array of detected blob. ndarray should be 4d. output matched ndarray index into new array
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
                    self.width = (xB-xA)
                    self.height = (yB-yA)
                    #resize threshold to be average of previous threshold and current threshold
                    self.xthreshold = ((self.xthreshold)*9+(xB-xA)/3)/10
                    self.ythreshold = ((self.ythreshold)*9+(yB-yA)/3)/10
                    #---blob no longer occluded as of this frame
                    self.occluded = False
                    self.occlusionframes = 0
                    #update the expected location for next frame
                    self.update()
                    #unique match
                    out[i] = 1
                #else the person is probably occluded
                else:
                    self.occluded = True
                    self.update()
    #remove blob from being present in frame. Ideally should only occur when the extrapolated position exceeds frame boundaries(edge of blob's bounding box hits edge of frame)
    #should be called every frame
    def check(self,frame, temp):
        self.wait = self.wait - 1
        #return if person not present. personally i want this false statement instead of the execute if true for some reason
        if self.present == False or self.needstest == False:
            return
            #set self.present to false once person leaves frame
        #check if blob is actually a person
        if self.needstest == True and self.wait <=0:
            self.wait = self.rewait
            #bounding box of blob with some extra padding (max 20 on each side)
            xA = int(self.xnow-(self.width)/2) - 20
            if (xA - 20) <= 0:
                xA = 0
            xB = int(self.xnow+(self.width)/2) + 20
            if (xB + 20) >= frame.shape[1]:
                xB = frame.shape[1]
            yA = int(self.ynow-(self.height)/2) - 20
            if  (yA - 20) <= 0:
                yA = 0
            yB = int(self.ynow+(self.height)/2) + 20
            if (yB + 20) >= frame.shape[0]:
                yB = frame.shape[0]
            temp = frame[yA:yB, xA:xB]
            #only run ped test if blob size is large enough
            if (yB-yA) > 50 or (xB-xA) > 20:
                #---detect pedestirans, make ndarray rects that holds top left corner and width and height of bounding box
                (rects, weights) = hog.detectMultiScale(temp, winStride=(4, 4), padding=(8, 8), scale=1.05)
                #---reorganize rects to hold top left and bottom right corner of bounding box (don't bother with nonmax suppression)
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                #ped detected, else do nothing
                if len(rects) > 0:
                    self.checkframesrequired = self.checkframesrequired - 1
                    if self.checkframesrequired <= 0:
                        self.ispedestrian = True
                        self.needstest = False
            #if ped was not detected in time
            if (self.framespresent > 500) and (self.checkframesrequired > 0):
                #within the first 10 frames of appearing, the blob did not have a ped present. no longer need to check this blob to see if is ped
                self.needstest = False
                #remove from play
                self.present = False
                self.occluded = True
        else:
            if ((self.xexpected+(self.xthreshold)) >= frame.shape[1]) or ((self.xexpected-(self.xthreshold)) <= 0) or ((self.yexpected+(self.ythreshold)) >= frame.shape[0]) or ((self.yexpected+(self.ythreshold)) <= 0):
                self.present = False
                self.needstest = False
            #---if occluded for too long, remove from being present. most likely false positive.
            #FOR NOW REMOVE BLOB IF OCCLUDED FOR TOO LONG
            #elif self.occlusionframes > 15:
                #self.present = False
                #self.occluded = True

#-------------------------------------------------
#---Initialize Pedestrian Variables
#--array to hold unique pedestrians. initialize with a false pedestrian to reduce iteration times later (explain later? i forgot why)
upedList=[]
upedList.append(Blob(0,0,0,0,0))
upedList[0].identity = "null" #shouldve been done automatically but w.e.
upedList[0].ispresent = False
upedList[0].needstest = False
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
minblobsize = 3000
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
cap = cv2.VideoCapture(0)
#---set brightness to reduce autowhite balance. Short training time in beginning should fix that issue a bit too
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(15,-5)
#---font for text
font = cv2.FONT_HERSHEY_SIMPLEX
#--counter for total number of frames. also used for current frame count
time = 0
#--frame min width
mwidth = 800

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
    thresh = cv2.threshold(frameDelta, 22, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours on thresholded image(iterations originally = 2)
    thresh = cv2.dilate(thresh, None, iterations=6)
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
        #temporary frame to cheeck for pedestrian
        temp = []
        ublobsList[i].check(frame, temp)        
    #check for unique blob atatchments/matches (as long as they didn't move too fast) in frame
    for i in range(len(testing)):
        if testing[i] == 0:
            #should only make new blob if the blob is actual pedestrian
            #test iftrue. may need to make another array for those
            what = what + 1
            ublobsList.append(Blob(what, blobs[i,0],blobs[i,1],blobs[i,2],blobs[i,3]))
    #add new unique persons to persons list
    for i in range(len(ublobsList)):
        if ublobsList[i].beenadded == False and ublobsList[i].ispedestrian == True:
            who = who + 1
            ublobsList[i].beenadded = True
            #nped = ublobsList[i]
            #nped.identity = str(who)
            ublobsList[i].identity = str(who)
            upedList.append(ublobsList[i])
    
    #-------------make a separate container for present unique pedestrians later? NOTED
    #print extrapolation rectangle
    for i in range(len(ublobsList)):
        if ublobsList[i].present == True:
            if ublobsList[i].identity != "null":#ispedestrian == True
                cv2.putText(frame, ublobsList[i].identity, (int(ublobsList[i].xnow), int(ublobsList[i].ynow)), font, 1, (255,255,255),1)
                cv2.rectangle(frame, (int(ublobsList[i].xexpected - ublobsList[i].xthreshold),int(ublobsList[i].yexpected - ublobsList[i].ythreshold)), (int(ublobsList[i].xexpected + ublobsList[i].xthreshold),int(ublobsList[i].yexpected + ublobsList[i].ythreshold)), (255, 0, 0), 2) 
            #show threshold box
            #elif ublobsList[i].ispedestrian == False:
                #cv2.rectangle(frame, (int(ublobsList[i].xexpected - ublobsList[i].xthreshold),int(ublobsList[i].yexpected - ublobsList[i].ythreshold)), (int(ublobsList[i].xexpected + ublobsList[i].xthreshold),int(ublobsList[i].yexpected + ublobsList[i].ythreshold)), (255, 0, 255), 2)
    #show the frame with bounding boxes
    cv2.imshow("frame", frame)

    backend = frame.copy()
    #---display min-suppressed bounding boxes (no small boxes should appear in any larger ones)
    for (xA, yA, xB, yB) in blobs:
        cv2.rectangle(backend, (xA, yA), (xB, yB), (0, 255, 0), 2)
    for i in range(len(ublobsList)):
        if ublobsList[i].identity == "null":
            cv2.putText(backend, ublobsList[i].identity, (int(ublobsList[i].xnow), int(ublobsList[i].ynow)), font, 1, (255,255,255),1)
            cv2.rectangle(backend, (int(ublobsList[i].xexpected - ublobsList[i].xthreshold),int(ublobsList[i].yexpected - ublobsList[i].ythreshold)), (int(ublobsList[i].xexpected + ublobsList[i].xthreshold),int(ublobsList[i].yexpected + ublobsList[i].ythreshold)), (255, 0, 255), 2) 

    
    # draw the timestamp on the frame
    #cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
    #show the frame with bounding boxes
    cv2.imshow("frame", frame)
    #show threshold box
    cv2.imshow("Thresh", thresh)
    #show difference between background and foreground
    cv2.imshow("Frame Delta", frameDelta)

    #show back end calculations
    cv2.imshow("Back End", backend)





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
