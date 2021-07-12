import cv2 as cv
import os
import numpy as np 
import dlib

detector = dlib.get_frontal_face_detector() #dlib detector gives face coords

predictor = dlib.shape_predictor("face_recognition\shape_predictor_68_face_landmarks.dat")


mood = input("Enter your mood: ")
frames = []
outputs = []

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # we use gray images to detect face and landmarks
    faces = detector(gray) # Using the detector on gray image 
    
    for face in faces:
        landmarks = predictor(gray,face)
        # print(landmarks.parts())

        expression = np.array([[point.x -  face.left(),point.y - face.top()] for point in landmarks.parts()[17:]]) #expression part
        # for point in landmarks.parts()[17:]:
        #    cv.circle(frame,(point.x,point.y),2,(255,0,0),3)


    if ret:
        cv.imshow("My screen",frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('c'):
        # f_path = name+'.jpg'
        # cv.imwrite(os.path.join('face_recognition/captures',f_path),frame)
        frames.append(expression.flatten())
        outputs.append([mood])

X = np.array(frames) #[[-------],[--------]] 10,000  -> 2
y = np.array(outputs) #[[omkar],[anuj]] 2

data = np.hstack([y,X]) # [[omkar],[---------],
                        #  [anuj],[---------]]

f_name = "face_mood.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old,data]) # adding new data (of expressions) each time we take new pics.

np.save(f_name,data)


cap.release()
cv.destroyAllWindows()
