#!/usr/bin/python -i
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show(im):
    cv2.imshow('detected circles',im)
    cv2.waitKey(0)

path = "image.jpg"

# Read image and convert to gray scale
im_in = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Crop the image to 1/4
#im_in = im_in[0:width/2, 0:height/2]

# Threshold: Set values equal to or above 220 to 0. Set values below 220 to 255.
# This convert the image to black and white.
th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling. Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255)

# Invert floodfilled image
im_floodfill_in = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_in


# Width and Height
width, height = im_in.shape

# Center Positions
x0 = width/2; y0 = height/2

# Radius (in pixels)
radius=10

for i in range(x0 - radius):
    
    # This dictionary stores the Total and Filled coordinates of the pixels.
    # This is used to calculate the pixels densisty
    index={"Total":[],"Filled":[]}
    
    # This 2 loops go through all the pixels of the image.
    # The euclidian distance is calculated in relation to the center pixel.
    # Then, this distance is compared with the radius defined.
    # Has a tolerance in this comparison of +- 5%.
    # The final if look if the pixel are black (filled pixel) and append
    # it in index["Filled"] if true.
    for x in range(0, width):
        for y in range(0, height):
            euclidian = np.sqrt((x - x0)**2 + (y - y0)**2) # Euclidian distance
            if radius*0.99 <= euclidian <= radius*1.01:
                index["Total"].append((x,y))
                if im_out[x,y] == 0:
                    index["Filled"].append((x,y))
    
    # The density of the pixels are:
    density = float(len(index["Filled"]))/float(len(index["Total"]))
    if density > 0.8:
        print("The radius is "+str(radius)+" and the pixels densisty are "+str(density))
        break
    
    # Increment the radius
    radius += i

for i in index:
    im_out[i]=0

# Create a mask
img_mask = np.full((height,width), 255)
for i in index["Filled"]:
    img_mask[i]=0

# Plot the arrow that represents the radius
cv2.arrowedLine(
    img = img_mask,
    pt1 = ( width, height ),
    pt2 = ( width, radius ),
    color = ( 0, 0, 0 ),
    thickness = 3,
    line_type = 5,
    tipLength = 0.05
)

f = plt.figure()
f.suptitle("Raio com preenchimento de pelo 80%: "+str(radius)+"(Valor em Pixels)", fontsize=16)

ax = plt.subplot(2,2,1)
ax.set_title("Imagem Original em Tons de Cinza")
ax.imshow(im_in, cmap='gray')

ax = plt.subplot(2,2,3)
ax.set_title("Imagem em Preto e Branco, Invertida")
ax.imshow(im_out, cmap='gray')

ax = plt.subplot(2,2,4)
ax.set_title("Pixels no Raio e Preenchimento")
ax.imshow(img_mask, cmap='gray')

plt.show(block=True)

cv2.imwrite( "raio.jpg", im_in );


