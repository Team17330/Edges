import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('pun.jpg',0)

canny = cv2.Canny(image,100,300)

sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=11)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=11)  
sobelxy = cv2.Sobel(image,cv2.CV_64F,1,1,ksize=11)  

img = cv2.GaussianBlur(image,(5,5),0) 
laplacian = cv2.Laplacian(img,cv2.CV_64F)

plt.subplot(3,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(sobelxy,cmap = 'gray')
plt.title('Sobel X Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.show()
