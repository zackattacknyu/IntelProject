
from keras.models import Sequential, Model

import matplotlib.image as mpimg
import numpy as np
import os

from keras.applications.vgg19 import VGG19
from scipy.misc import imresize
import cv2

wholeVGGnetwork = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
vggNetwork4096Output = Model(input=wholeVGGnetwork.input,output=wholeVGGnetwork.get_layer('fc2').output)
numFeatures = vggNetwork4096Output.output_shape[1]

trainDirectory = 'train'
testDirectory = 'test'
typeDirNames = ['Type_1','Type_2','Type_3']
vggImgSizeChannel = (224,224)
vggImgSize = (1,3,224,224)

trainDirectories = [os.path.join(trainDirectory,typeNm) for typeNm in typeDirNames]

def obtainJPGfileList(dirName):
    curFiles = []
    for fileNm in os.listdir(dirName):
        if (fileNm.endswith(".jpg")):
            curFiles.append(fileNm)
    return curFiles

print("Reading training files")
trainFiles = []
numFilesEachType = []
for dirName in trainDirectories:
    curFiles = obtainJPGfileList(dirName)
    numFilesEachType.append(len(curFiles))
    trainFiles.append(curFiles)

print("Reading testing files")
testFiles = obtainJPGfileList(os.path.join(testDirectory))
#np.save('XTestFileNames.npy',testFiles)

numTestFiles = len(testFiles)

numTotalTrainFiles = np.sum(numFilesEachType)

XTrain = np.zeros((numTotalTrainFiles,numFeatures))
YTrain = np.zeros((numTotalTrainFiles))
XTest = np.zeros((numTestFiles,numFeatures))

def getVGGoutput(currentFilePath):
    currentCervixImage = mpimg.imread(currentFilePath)
    resizedCervixImage = np.zeros(vggImgSize)
    for jj in range(3):
        img = currentCervixImage[:, :, jj]
        imgResized = cv2.resize(img, vggImgSizeChannel)
        resizedCervixImage[0,jj, :, :] = imgResized
    return vggNetwork4096Output.predict(resizedCervixImage)

print("Now generating features for each training file")
ind = 0
for typeI in range(3):
    trainFilesCurType = trainFiles[typeI]
    for fileNm in trainFilesCurType:
        print('Generating features for file ' + str(ind+1) + ' of ' + str(numTotalTrainFiles))
        currentFilePath = os.path.join(trainDirectory,typeDirNames[typeI],fileNm)
        print(currentFilePath)
        XTrain[ind,:] = getVGGoutput(currentFilePath)
        YTrain[ind] = typeI
        ind = ind + 1

print("Now generating features for each testing file")
ind = 0
for fileNm in testFiles:
    print('Generating features for file ' + str(ind+1) + ' of ' + str(numTestFiles))
    currentFilePath = os.path.join(testDirectory,fileNm)
    XTest[ind,:] = getVGGoutput(currentFilePath)
    ind = ind + 1


np.save('xTrain4096Output.npy',XTrain)
np.save('xTest4096Output.npy',XTest)
np.save('yTrain.npy',YTrain)

