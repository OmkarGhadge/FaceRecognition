import cv2 as cv

cap = cv.VideoCapture(0)


detector = cv.CascadeClassifier('face_detection\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret:
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = frame[y:y+h,x:x+w]

        fix = cv.resize(cut,(200,200))
        cv.imshow("my screen",frame)
        cv.imshow("my cut",fix)
        
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

