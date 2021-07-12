import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

DIR = r'C:/Users/omkar/Desktop/opencv/Resources/Faces/train'

haar_cascade = cv.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')

features = [] # image arrays of the faces
labels = [] # label for each face acc to list -> Ben afflek - 0 etc.

def create_train():
    for person in people:
        path = os.path.join(DIR,person) #path of the folders -eg: C:\Users\omkar\Desktop\opencv\Resources\Faces\train/Ben Afflek
        label = people.index(person)

        for img in os.listdir(path): #iterating over all images in a folder
            img_path = os.path.join(path,img) # a particular image

            img_arr = cv.imread(img_path)
            gray = cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)

            #DETECTION OF THE IMAGE USIGN HAARCASCADE:
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            #LOOPING OVER EVERY FACE:
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w] # REGION OF INTEREST
                features.append(faces_roi)
                labels.append(label)

            
create_train()
# print(f'Length of the features is = {len(features)}')
# print(f'Length of the labels is = {len(labels)}')

print('Training done--------------------')

face_recognizer = cv.face.LBPHFaceRecognizer_create() #INSTANTIATING THE RECOGNIZER

#TRAINING THE RECOGNIZER on feature and label list.
features=np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)

