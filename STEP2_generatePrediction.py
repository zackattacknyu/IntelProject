
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
import numpy as np
from sklearn import cross_validation

Xtrain = np.load('xTrain4096Output.npy')
Xtest = np.load('xTest4096Output.npy')
Ytrain = np.load('yTrain.npy')
input_dim = 4096

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    Xtrain, Ytrain, random_state=42, stratify=Ytrain, test_size=0.20)

trn_yOneHot = np_utils.to_categorical(trn_y, 3)
val_yOneHot = np_utils.to_categorical(val_y, 3)

modelUse = Sequential()
modelUse.add(Dense(512,init='normal',activation='relu',input_dim=input_dim))
modelUse.add(Dense(32,init='normal',activation='sigmoid'))
modelUse.add(Dense(3,init='normal',activation='softmax'))
modelUse.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Now fitting Neural Network to cervix shape")
modelUse.fit(trn_x, trn_yOneHot, batch_size=500, nb_epoch=30,
                  verbose=1, validation_data=(val_x, val_yOneHot))

YHatTest = modelUse.predict(Xtest)

