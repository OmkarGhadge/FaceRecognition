import cv2 as cv
import numpy as np

img  = cv.imread('data\cats.jpg') #matrix of pixels
cats_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('og',cats_gray)

# LAPLACIAN: -> coverts to gradients.
lap = cv.Laplacian(cats_gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('lap',lap)

# SOBEL -  gradient in x and y dirents

sobelx = cv.Sobel(cats_gray,cv.CV_64F,1,0)
sobely = cv.Sobel(cats_gray,cv.CV_64F,0,1)
# cv.imshow('sx',sobelx)
# cv.imshow('sy',sobely)

#combined:
comb = cv.bitwise_or(sobelx,sobely)
cv.imshow('combined',comb)

#CANNY:
can = cv.Canny(cats_gray,150,175)
cv.imshow('canny',can)

cv.waitKey(0)