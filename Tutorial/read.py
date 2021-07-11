import cv2 as cv

#reading image:
img  = cv.imread('data\cat.jpg') #matrix of pixels
#cv.imshow('Catt',img)
#cv.waitKey(0)

#reading video:
capture = cv.VideoCapture('data\dog.mp4') # 0 - for webcam
while True:
    isTrue, frame = capture.read()
    cv.imshow('vid',frame) #showing frame by frame

    if cv.waitKey(20) & 0xFF==ord('d'): #breaks out of loop stops the video when d key is pressed
        break
capture.release()
cv.destroyAllWindows() 



