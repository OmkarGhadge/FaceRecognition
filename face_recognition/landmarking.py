import cv2 as cv
import os
import numpy as np 
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("face_recognition\shape_predictor_68_face_landmarks.dat")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # we use gray images to detect face and landmarks
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray,face)
        # print(landmarks.parts())
        nose = landmarks.parts()[27]
        lip_up = landmarks.parts()[62].y
        lip_down = landmarks.parts()[66].y

        if (lip_down - lip_up) > 5:
            print("open")
        else:
            print("close")
        for point in landmarks.parts()[48:]: # only mouth related

           cv.circle(frame,(point.x,point.y),2,(255,0,0),3)


    if ret:
        cv.imshow("My screen",frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
