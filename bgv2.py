#bgv2
# import the necessary packages
from __future__ import print_function
from imutils.video import count_frames
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import datetime
import imutils
import time
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture('last0.mp4')
	camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
         
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
bl = []
a = []
blobs=[]
# loop over the frames of the video
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	blob=[]
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=min(400, frame.shape[1]))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image. originally 2 iterations
	thresh = cv2.dilate(thresh, None, iterations=5)
	(im2,cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
	# loop over the contours and place ones that pass into array
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
                #temp = nd.array([x, y, w, h])
		blob.append([x, y, w, h])
	#ERROR THAT OFTEN OCCURS REQUIRING TABIFYING OF REGION IN CODE FOR SOME REASON. STUPID
	blobs = np.array([[c[0], c[1], (c[0] + c[2]), (c[1] + c[3])] for c in blob])
        #---keep overlapping boxes using large overlap threshold(1) or don't allow overlapping with (0)
	blobs = non_max_suppression(blobs, probs=None, overlapThresh=.01)
        #---display min-suppressed bounding boxes (no small boxes should appear in any larger ones)
	for (xA, yA, xB, yB) in blobs:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # draw the timestamp on the frame
	#cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
	# show the frame
	bl = len(blob)
        
	cv2.imshow("frame", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `Esc` key is pressed, break from the lop
	if key == 27:
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
#----backround sub + blob highlighting
#http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/


#---bounding box thresholding


#---Camera parameters to lock brightness
#http://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python

#adding array holding bounding boxes of contours in the same fashion as rects in frames_with_people_with_class
