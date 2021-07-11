import cv2 as cv
img = cv.imread('data\park.jpg')


# BGR TO GRAY SCALE
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray[200:400,400:500])


# BGR TO HSV
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow('hsv',hsv)

# Other color spaces: L a b


#NOTE-> OPENCV USES BGR FORMAT WHILE MATPLOTLIB ASSUMES ITS RGB
# HENCE WE HAVE TO CONVERT BGR TO RGB:
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
rows=1
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(img)

fig.add_subplot(rows, columns, 2)
converted = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(converted)

plt.show()


cv.waitKey(0)