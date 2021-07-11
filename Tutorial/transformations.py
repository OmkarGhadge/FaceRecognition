import cv2 as cv
import numpy as np

park = cv.imread('data\park.jpg')
# TRANSLATION - up/down/left/right
def translate(img,x,y):
    trans_Mat = np.float32([[1,0,x],[0,1,y]])
    dim = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, trans_Mat,dim)

translated = translate(park,100,100)   # -x means left and +y means down ....
# cv.imshow('ts',translated)


#ROTATION:
def rotate(img, angle, rotpt=None):
    (h,w) = img.shape[:2]

    if rotpt is None:
        rotpt = (w//2,h//2)

    rotMat = cv.getRotationMatrix2D(rotpt,angle,1)
    dim = (w,h)

    return cv.warpAffine(img,rotMat,dim)

rot = rotate(park,45)
# cv.imshow('rotated',rot)

# FLIPPING:
flip  = cv.flip(park,1) # 0->vertically 1-> hz, -1-> both
cv.imshow('f',flip)

cv.waitKey(0)