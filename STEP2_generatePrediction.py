import numpy as np
import os
import scipy.io as sio
import numpy.matlib
from scipy.ndimage.interpolation import zoom
import numpy as np
import os
#import dicom
#import glob
from matplotlib import pyplot as plt
import os
import csv
import cv2
import datetime
# import mxnet as mx
# import pandas as pd
# from sklearn import cross_validation
# import xgboost as xgb
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from scipy.ndimage.interpolation import zoom
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import csv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import random

from keras.applications.vgg19 import VGG19
import scipy.io as sio
from scipy.misc import imresize


wholeVGGnetwork = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
vggNetwork4096Output = Model(input=wholeVGGnetwork.input,output=wholeVGGnetwork.get_layer('fc2').output)
numFeatures = vggNetwork4096Output.output_shape[1]

trainDirectory = 'train'
typeDirNames = ['Type_1','Type_2','Type_3']
vggImgSizeChannel = (224,224)
vggImgSize = (3,224,224)

trainDirectories = [os.path.join(trainDirectory,typeNm) for typeNm in typeDirNames]

print("Reading training files")
trainFiles = []
numFilesEachType = []
for dirName in trainDirectories:
    curFiles = []
    for fileNm in os.listdir(dirName):
        if(fileNm.endswith(".jpg")):
            curFiles.append(fileNm)
    numFilesEachType.append(len(curFiles))
    trainFiles.append(curFiles)

numTotalTrainFiles = np.sum(numFilesEachType)

XTrain = np.zeros((numTotalTrainFiles,numFeatures))
YTrain = np.zeros((numTotalTrainFiles))

print("Now generating features for each training file")
ind = 0
for typeI in range(3):
    trainFilesCurType = trainFiles[typeI]
    for fileNm in trainFilesCurType:
        print('Generating features for file ' + str(ind+1) + ' of ' + str(numTotalTrainFiles))
        currentFilePath = os.path.join(trainDirectory,typeDirNames[typeI],fileNm)
        currentCervixImage = mpimg.imread(currentFilePath)
        resizedCervixImage = np.zeros(vggImgSize)
        for jj in range(3):
            img = currentCervixImage[:,:,jj]
            imgResized = imresize(img,vggImgSizeChannel)
            resizedCervixImage[jj,:,:] = imgResized
        XTrain[ind,:] = vggNetwork4096Output.predict(resizedCervixImage)
        YTrain[ind] = typeI

"""
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
"""