#!/usr/bin/env python
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import dlib
import numpy as np
 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Read Image

cap = cv2.VideoCapture(0)

while(True):

    ret, im = cap.read();
    im = cv2.resize(im, (0,0), fx=0.3, fy=0.3)
    size = im.shape

    dets = detector(im, 1)

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        global shape 
        shape = predictor(im, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
         
        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                (shape.part(30).x, shape.part(30).y),     # Nose tip
                                (shape.part(8).x, shape.part(8).y),     # Chin
                                (shape.part(36).x, shape.part(36).y),     # Left eye left corner
                                (shape.part(45).x, shape.part(45).y),     # Right eye right corne
                                (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
                                (shape.part(54).x, shape.part(54).y)      # Right mouth corner
                            ], dtype="double")
     
        # 3D model points.
        model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
     
    # Camera internals
        # 3D model points.
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
         
        print "Camera Matrix :\n {0}".format(camera_matrix)
    
     
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
         
        print "Rotation Vector:\n {0}".format(rotation_vector)
        print "Translation Vector:\n {0}".format(translation_vector)
         
         
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
         
         
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
         
        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
         
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
         
        cv2.line(im, p1, p2, (255,0,0), 2)
     
    # Display image
    cv2.imshow("Output", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()