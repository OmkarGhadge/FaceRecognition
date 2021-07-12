import cv2 as cv
import os
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

cap = cv.VideoCapture(0)

detector = cv.CascadeClassifier('face_detection\haarcascade_frontalface_default.xml')

name = input("Enter your name: ")

frames = []
outputs = []
while True:
    ret, frame = cap.read()
    if ret:
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = frame[y:y+h,x:x+w]

            fix = cv.resize(cut,(100,100))
            gray = cv.cvtColor(fix,cv.COLOR_BGR2GRAY) #less features
        cv.imshow("my cut",gray)
        cv.imshow("my screen",frame)
        
    key = cv.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('c'):
        # f_path = name+'.jpg'
        # cv.imwrite(os.path.join('face_recognition/captures',f_path),frame)
        frames.append(gray.flatten())
        outputs.append([name])

X = np.array(frames) #[[-------],[--------]] 10,000  -> 2
y = np.array(outputs) #[[omkar],[anuj]] 2

data = np.hstack([y,X]) # [[omkar],[---------],
                        #  [anuj],[---------]]

model = KNeighborsClassifier()
model.fit(X,y)

f_name = 'face.npy'
if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old,data])


np.save(f_name,data)

cap.release()
cv.destroyAllWindows()
