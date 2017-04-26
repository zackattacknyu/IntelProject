
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
import numpy as np
from sklearn import cross_validation
import time
import datetime
import csv
import xgboost as xgb

Xtrain = np.load('xTrain4096Output.npy')
Xtest = np.load('xTest4096Output.npy')
Ytrain = np.load('yTrain.npy')
input_dim = 4096

def getBinaryArray(array,num):
    return (array == num).astype('int')


trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    Xtrain, Ytrain, random_state=42, stratify=Ytrain, test_size=0.20)

trn_yOneHot = np_utils.to_categorical(trn_y, 3)
val_yOneHot = np_utils.to_categorical(val_y, 3)

trn_y1 = getBinaryArray(trn_y,0)
trn_y2 = getBinaryArray(trn_y,1)
trn_y3 = getBinaryArray(trn_y,2)
val_y1 = getBinaryArray(val_y,0)
val_y2 = getBinaryArray(val_y,1)
val_y3 = getBinaryArray(val_y,2)

modelUse = Sequential()
modelUse.add(Dense(1024,init='normal',activation='relu',input_dim=input_dim))
modelUse.add(Dense(32,init='normal',activation='sigmoid'))
modelUse.add(Dense(3,init='normal',activation='softmax'))
modelUse.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print("Now fitting Neural Network to cervix shape")
modelUse.fit(trn_x, trn_yOneHot, batch_size=500, nb_epoch=30,
                  verbose=1, validation_data=(val_x, val_yOneHot))

def getPrediction(trainY,validationY):
    clf = xgb.XGBRegressor(max_depth=15,
                               n_estimators=1500,
                               min_child_weight=9,
                               learning_rate=0.05,
                               nthread=8,
                               subsample=0.80,
                               colsample_bytree=0.80,
                               seed=4242)
    clf.fit(trn_x, trainY, eval_set=[(val_x, validationY)], verbose=True,
            eval_metric='logloss', early_stopping_rounds=100)
    return clf.predict(Xtest,output_margin=True)

fileNames = np.load('XTestFileNames.npy')
YHatTestXGB = np.zeros((len(fileNames),3))
YHatTestXGB[:,0] = getPrediction(trn_y1,val_y1)
YHatTestXGB[:,1] = getPrediction(trn_y2,val_y2)
YHatTestXGB[:,2] = getPrediction(trn_y3,val_y3)
YHatTestXGBSums=np.reshape(np.sum(YHatTestXGB,axis=1),(YHatTestXGB.shape[0],1))
YHatTestXGBOutput = YHatTestXGB/YHatTestXGBSums


YHatTestNN = modelUse.predict(Xtest)

def obtainPred(origPred):
    if(origPred<0):
        return 0
    elif(origPred>1):
        return 1
    else:
        return origPred

def writeKagglePredictionFile(prefixString,pred):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    fileName = prefixString + st + '.csv'

    with open(fileName, 'w') as csvfile:
        fieldnames = ['image_name', 'Type_1','Type_2','Type_3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ind in range(len(fileNames)):
            writer.writerow({'image_name': fileNames[ind],
                             'Type_1': str(obtainPred(pred[ind,0])),
                             'Type_2': str(obtainPred(pred[ind,1])),
                             'Type_3': str(obtainPred(pred[ind,2]))})

prefixString = 'submissions/InitNeuralNetPrediction_'
writeKagglePredictionFile(prefixString, YHatTestNN)

prefixString = 'submissions/XGBoostPrediction_'
writeKagglePredictionFile(prefixString, YHatTestXGBOutput)

prefixString = 'submissions/XGB_NN_EnsemblePrediction_'
writeKagglePredictionFile(prefixString, (YHatTestXGBOutput+YHatTestNN)/2)