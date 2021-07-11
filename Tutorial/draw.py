import cv2 as cv
import numpy as np


blank = np.zeros((500,500,3),dtype='uint8')
# cv.imshow('imag',blank)

# 1. Paint the image a color
# blank[100:200,100:300] = 0,255,0 # B G R VALUES FOR PIXELS OF GIVEN DIMENSTION.
# cv.imshow('green',blank) 

# 2. Draw a rectangle
# cv.rectangle(blank,(0,0),(250,250),(255,255,0),thickness = 2)
# cv.imshow('rect',blank)

# 3. Circle
# cv.circle(blank,(blank.shape[0]//2,blank.shape[1]//2),40,(0,0,255),-1)
# cv.imshow('circ',blank)

# 4. Text
# cv.putText(blank,'ZODIAC',(225,225),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
# cv.imshow('zod',blank)


img  = cv.imread('data\cat.jpg') #matrix of pixels
# cv.imshow('Catt',img)


# 5. Converting to gray scale
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Catt',gray)

# 6. Blur
blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
# cv.imshow('Catt',blur)

# 7. Edge Cascade
# canny = cv.Canny(img,125,180)
# canny_b = cv.Canny(blur,125,180)
# cv.imshow('can',canny)
# cv.imshow('blur_can',canny_b) #edges are less 

# More functions are DILATING AND ERODING.

# 8. RESIZE :
# interpolation:
# a. INTER_AREA - when you're shrinking img to smaller dimensions.
# b. INTER_LINEAR/CUBIC - scaling imag to much larger dimension.
park = cv.imread('data\park.jpg')
cv.imshow('org',park)
rs = cv.resize(park,(200,250),interpolation=cv.INTER_CUBIC)
cv.imshow('rs',rs)

# 9. CROPPING:
cropped = park[200:400,50:200]
cv.imshow('ci',cropped)


cv.waitKey(0)


