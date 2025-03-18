#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:44:03 2024

@author: getasewalemu
"""
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop,Adam 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
import numpy as np
  
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mammographic_masses.data.txt', na_values = '?')
df.columns = ['BI_RADS', 'age', 'shape', 'margin', 'density','severity']
df = df.dropna()
df.reset_index(inplace=True)
scaler = StandardScaler()
x = df[['BI_RADS', 'age', 'shape', 'margin', 'density']]
y = df['severity']
x = scaler.fit_transform(x)
def train_testsplit(x, y, test_ratio):
    test_size = int((test_ratio)* df.shape[0])
    #test_size = test_ratio*len(x)
    random_ind = np.random.permutation(df.shape[0])
    x_train = x[random_ind[test_size:]]
    y_train = y[random_ind[test_size:]]
    x_test = x[random_ind[:test_size]]
    y_test = y[random_ind[:test_size]]
    return x_train, y_train,x_test,y_test
    
    
x_train, y_train, x_test, y_test = train_testsplit(x, y, test_ratio = 0.2)
print(x_train)

def neural_network():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(5,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = neural_network()
model.compile(loss='binary_crossentropy',
              #optimizer=RMSprop(),
              optimizer=Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(x_test, y_test))

'''

def knn_model(k):
    knn = KNeighborsClassifier(n_neighbors=k) 
      
    return knn
    
knn = knn_model(2)
knn.fit(x_train, y_train)
print(knn.predict(x_test))
''' 