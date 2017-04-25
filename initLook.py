import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import random

trainDirectory1 = os.path.join('train','Type_1')
trainImgs = os.listdir(trainDirectory1)

numRows = 4
channelNames = ['Red ','Green ','Blue ']

for rowN in range(numRows):
    randomIndex = (np.floor(random.random()*len(trainImgs))).astype('int')
    sampleImg = os.path.join(trainDirectory1,trainImgs[randomIndex])

    imgRead = mpimg.imread(sampleImg)
    plt.subplot(numRows,4,1 + rowN*4)
    plt.imshow(imgRead)
    if(rowN<1):
        plt.title('Original Image')
    for chanNum in range(3):
        plt.subplot(numRows,4,chanNum+2 + rowN*4)
        if (rowN < 1):
            plt.title(channelNames[chanNum] + 'Channel Only')
        plt.imshow(imgRead[:,:,chanNum])
        plt.colorbar()

plt.show()
