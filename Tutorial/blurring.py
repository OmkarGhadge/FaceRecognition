import cv2 as cv

img = cv.imread('data\Cats.jpg')


# Avergaing Blur: (middle pixel of kernel is avg of surr pixels)
avg = cv.blur(img,(3,3)) # more the kernel size-> more the BLURRR
cv.imshow('avg',avg)

# Gaussian Blur: the pixels nearest the center of the kernel are given more weight than those far away from the center.WEIGHTED AVERAGE
gaus = cv.GaussianBlur(img,(3,3),sigmaX=0)
cv.imshow('gaus',gaus)

#Median Blurr : median instead of average -> effective in NOISE removal.
med = cv.medianBlur(img,3)
cv.imshow('med',med)

# Bilateral Blurr: Most effective -> RETAINS THE EDGES

bi = cv.bilateralFilter(img,5,15,15)
cv.imshow('bila',bi)

cv.waitKey(0)