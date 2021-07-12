import numpy as np 
import cv2 as cv

people = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
haar_cascade = cv.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\omkar\Desktop\opencv\Resources\Faces\val\mindy_kaling\2.jpg')
def test_on(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #Detect:
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w] # REGION OF INTEREST
    #Prediction:
        label,confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {label} with a confidence of {confidence}')

        cv.putText(img, str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('Detected Face',img)

test_on(img)
cv.waitKey(0)


#NOT THAT GOOD A FACE RECOGNITION :(