from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean



def threshold(imageArray):
    balanceAr = []
    newAr = imageArray
    for row in imageArray:
        for pixel in row:
            avgNum = mean(pixel[:3])
            balanceAr.append(avgNum)
    balance = mean(balanceAr)
    for row in newAr:
        for pixel in row:
            if mean(pixel[:3]) > balance:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255

            else:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0

    return newAr
i = Image.open("test_happy.jpg")

####i1 = Image.open("test_sad.jpg")

iar = np.array(i)
iar = threshold(iar)
fig = plt.figure()
ax = plt.subplot2grid((8,6),(0,0), rowspan = 4, colspan = 4)
ax.imshow(iar)

plt.show()


