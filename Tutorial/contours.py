import cv2 as cv
import numpy as np

# Counters are similar to edges.

cats = cv.imread('data\cats.jpg')
cats_gray = cv.cvtColor(cats,cv.COLOR_BGR2GRAY)
# cv.imshow('cats',cats_gray)

cats_canny = cv.Canny(cats_gray,125,175)
# cv.imshow('cats_can',cats_canny)

# contours, heirarchies = cv.findContours(cats_canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
# print("no of countours is ",len(contours)) # 2479 contours

# Now if we blur image and find no of contours we get 380 justttt.

# threshold  -> converrt to zero,black or white.
ret, thresh = cv.threshold(cats_gray,125,255, cv.THRESH_BINARY) # if pixel val is below 125 --> 0.
# cv.imshow('thrsh',thresh)

contours, heirarchies = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

print("no of countours is ",len(contours))  #839 contours/

# DISPLAYING CONTOURS:
blank = np.zeros(cats.shape,dtype='uint8')
cv.drawContours(blank, contours,-1, (0,0,255),1) #like edges
cv.imshow('cont draw',blank)

# generally, we use canny and then use contours .


cv.waitKey(0)

