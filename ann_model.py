# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:40:25 2024

@author: getch
"""
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop,Adam 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
from processing import preprocessing 
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
class NN_model:
    def neural_network():
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(5,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model
    def evaluate(self,y_test, pred):
        
        tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
        #print((tn, fp, fn, tp))
        precission = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*(recall*precission)/(recall+precission)
        acc = (tp+tn)/(tp+tn+fp+fn)
        print('accuracy:', acc*100)
        print("precission:" ,precission*100)
        print("recall:" ,recall*100)
        print("f1_score:" ,f1_score*100)

parse = argparse.ArgumentParser(description= "binary classification using ANN")
parse.add_argument('--input', type= str, help='please insert the input .txt file.')
args = parse.parse_args()
data = args.input
#data = preprocessing('mammographic_masses.data.txt')
data = preprocessing(data)
x, y = data.process() 
x_train, y_train,x_test,y_test = data.train_test_split(x, y, split_ratio = 0.2)
model = NN_model.neural_network()
model.compile(loss='binary_crossentropy',
              #optimizer=RMSprop(),
              optimizer=Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(x_test, y_test))
#model.evaluate(y_test, )

