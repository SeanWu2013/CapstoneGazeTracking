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
        self.xthreshold = (xB-xA)/5
        self.ythreshold = (yB-yA)/5
        #may need to count frames occluded
        self.occlusionframes = 0
        #---most likely a false positive if it keeps dropping.
        self.drops = 0
        self.framespresent = 0
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
                    self.xthreshold = ((self.xthreshold)+(xB-xA)/5)/2
                    self.ythreshold = ((self.ythreshold)+(yB-yA)/5)/2
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
            if self.occlusionframes > 15:
                self.present == False
                self.occluded == True
            #worry about deleting people if they get occluded for too long later




#---initialize variables
#array to hold unique pedestrians. initialize with a false pedestrian to reduce iteration times later
pedList=[]
pedList.append(Pedestrian(0,0,0,0,0))
pedList[0].identity = "0"
#new pedestrians detected / change in number of pedestrians
newpeds = 0
#unique pedestrian detector (should increase each time newped > 0)
who = 0
#counter for total number of frames. also used to for current frame count
totalf = 0

#---capture video from video source 'video.mp4' is video path
cap = cv2.VideoCapture('two1.mp4')
#---initialize histogram of oriented gradients
hog = cv2.HOGDescriptor()
#---initialize people detecting svm
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#---font for text
font = cv2.FONT_HERSHEY_SIMPLEX

#---main function
while True:
    (grabbed, image) = cap.read()
    #---check if there is still video
    if not grabbed:
        break
    #---iterate number of frames captured
    totalf += 1
    #---resize image for efficiency
    image = imutils.resize(image, width=min(500, image.shape[1]))
    
    #---detect pedestirans, make ndarray rects that holds top left corner and width and height of bounding box
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    #---reorganize rects to hold top left and bottom right corner of bounding box (i believe this was necessary for suppression)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    #---keep overlapping boxes using large overlap threshold(1) or don't allow overlapping with (0)
    rects = non_max_suppression(rects, probs=None, overlapThresh=.3)

    #--- current number of pedestriands detected = size of thresholded rectangles array
    pednum = len(rects)
    #array to hold 1/0 indicating which pedestrians in frame have a unique ped associated to them
    associated = [0]*len(rects)
    
    #--- check for unique pedestrians in frame
    for i in range(len(pedList)):
        #only check for predestrians that may still be present. currently program is expected to slow down as unique peds increase.
        if pedList[i].present == True:
            pedList[i].find(rects,associated)
    #add unique new pedestrians to list of unique pedestrians 
    for i in range(len(associated)):
        if associated[i] == 0:
            #---should only make new peds when they come in from edge of screen. Else, results in too many false positives. comment out if statement to see.
            #allow for new peds already standing in frame in beginning
            if totalf <= 10:
                who = who + 1
                pedList.append(Pedestrian(who,rects[i,0],rects[i,1],rects[i,2],rects[i,3]))
            #---new peds comming in from edge of screen
            if totalf > 10 and ((rects[i,0] < 20) or (rects[i,1] < 20) or (rects[i,2] > image.shape[1] - 20) or (rects[i,3] > image.shape[0] - 20)):
                who = who + 1
                pedList.append(Pedestrian(who,rects[i,0],rects[i,1],rects[i,2],rects[i,3]))
    #-------------make a separate container for present unique pedestrians later
    for i in range(len(pedList)):
        if pedList[i].present == True:
            pedList[i].checkleave(image)
            cv2.putText(image, pedList[i].identity, (int(pedList[i].xnow), int(pedList[i].ynow)), font, 1, (255,255,255),1)
            #show threshold box
            cv2.rectangle(image, (int(pedList[i].xexpected - pedList[i].xthreshold),int(pedList[i].yexpected - pedList[i].ythreshold)), (int(pedList[i].xexpected + pedList[i].xthreshold),int(pedList[i].yexpected + pedList[i].ythreshold)), (255, 0, 0), 2) 
    
    
    #--- draw bounding boxes
    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        #cv2.putText(image, str(1), (xA, yA), font, 1, (255,255,255), 2)
    #---Show Frame
    cv2.imshow("Frame", image)
    #---iterating through rects
    for i in range(len(rects)):
       # seems to output top left corner and bottom right corner of box
       print("rectangle: %s\n firstindex: %s\n index: %i " %(rects[i], rects[i,0], i))
    #--- press 'Esc' to exit program early
    print("unique people: %i\n" %(who))
    
    #---After Frame shown, can save some varialbes required for next frame (prepare for next frame)
    newped = 0
    
    #--- press 'Esc' to exit program early
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#--- Print results    
print ("total frames: %i" %(totalf))
#--- release video capture device from use
cap.release()
cv2.destroyAllWindows()
