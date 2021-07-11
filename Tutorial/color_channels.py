import cv2 as cv

img = cv.imread('data\park.jpg')

b,g,r = cv.split(img)

di = {'blue':b,'green':g,'red':r}

for i,j in di.items():
    # cv.imshow('{}'.format(i),j)
    #displayed as gray scale images

#merged
mergd = cv.merge([b,g,r])
# cv.imshow('m',mergd)

import numpy as np
#GETTING THE COLORS:
#make a blank image of same dimension (only first 2)
blank = np.zeros(img.shape[:2],dtype='uint8')
blue = cv.merge([b,blank,blank])




cv.waitKey(0)