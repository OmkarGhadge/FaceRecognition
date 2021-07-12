import cv2 as cv

img = cv.imread('Resources\Photos\lady.jpg')

img2 = cv.imread('Resources\Photos\group 1.jpg')


def detect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml') #trained model.

    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=1) #returns the rectangular coords.

    print(f"No of faces found = {len(faces_rect)}")

    #drawing the rectangle:
    for (x,y,w,h) in faces_rect:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('detected face',img)

# detect(img)
detect(img2)

# NOTE - increasing minNeighbours reduces the noise (less faces detected)
cv.waitKey(0)