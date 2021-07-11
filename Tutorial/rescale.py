import cv2 as cv

#reading image:
img  = cv.imread('data\cat.jpg') #matrix of pixels

def rescale_frame(frame, scale = 0.75):
    # Video, img and live video.
    w = int(frame.shape[1]*scale)
    h = int(frame.shape[0]*scale)
    dim = (w,h)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

#only for live video:
def change_res(w,h):
    capture.set(3,w)
    capture.set(4,h)


capture = cv.VideoCapture('data\dog.mp4') # 0 - for webcam
while True:
    isTrue, frame = capture.read()
    frame_resize = rescale_frame(frame)
    cv.imshow('vid',frame)
    cv.imshow('vid2',frame_resize) #showing frame by frame

    if cv.waitKey(20) & 0xFF==ord('d'): #breaks out of loop stops the video when d key is pressed
        break
capture.release()
cv.destroyAllWindows() 