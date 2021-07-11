import cv2 as cv

#THRESHOLDING -  BINARIZATION OF IMAGES
#converting the pixels to 0 or 255 acc to a set threshold.


cats = cv.imread('data\cats.jpg')
cats_gray = cv.cvtColor(cats,cv.COLOR_BGR2GRAY)

threshold, thresh = cv.threshold(cats_gray,150,255,cv.THRESH_BINARY) # 150 is the Thresh point

cv.imshow('simple threshold',thresh)


# ADAPTIVE THRESHOLDING:
#Automaticall finds threshol value

adap_thresh = cv.adaptiveThreshold(cats_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('adaptive threshold',adap_thresh)


cv.waitKey(0)