  
import cv2 as cv
import numpy as np
import dlib
from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_mood.npy")

print(data.shape, data.dtype)

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

detector = dlib.get_frontal_face_detector() #dlib detector gives face coords

predictor = dlib.shape_predictor("face_recognition\shape_predictor_68_face_landmarks.dat")

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

        print(model.predict([expression.flatten()]))

    if ret:
        cv.imshow("My screen",frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

    
cap.release()
cv.destroyAllWindows()
