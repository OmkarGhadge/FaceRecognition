import cv2 as cv
import numpy as np

#reading image:
img  = cv.imread('data\cats.jpg') #matrix of pixels

blank = np.zeros(img.shape[:2],dtype= 'uint8')

mask = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)

masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked',masked)


# WE CAN MAKE ANY SHAPE AND SIZES MASK WITH THIS BY BITWISE OPERATORS.
cv.waitKey(0)