#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:27:54 2024

@author: getasewalemu
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mammographic_masses.data.txt', na_values = '?')
df.columns = ['BI_RADS', 'age', 'shape', 'margin', 'density','severity']
params = {
            'max_depth': 2,
            'learning_rate': 0.5,
            'n_estimators':1000}
df = df.dropna()
df.reset_index(inplace=True)
scaler = StandardScaler()
x = df[['BI_RADS', 'age', 'shape', 'margin', 'density']]
y = df['severity']
x = scaler.fit_transform(x)
def train_testsplit(x, y, test_ratio):
    np.random.seed(10)
    test_size = int((test_ratio)* df.shape[0])
    #test_size = test_ratio*len(x)
    random_ind = np.random.permutation(df.shape[0])
    x_train = x[random_ind[test_size:]]
    y_train = y[random_ind[test_size:]]
    x_test = x[random_ind[:test_size]]
    y_test = y[random_ind[:test_size]]
    return x_train, y_train,x_test,y_test
     
x_train, y_train, x_test, y_test = train_testsplit(x, y, test_ratio = 0.2)
model = GradientBoostingClassifier(**params)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print((tn, fp, fn, tp))
precission = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(recall*precission)/(recall+precission)
acc = (tp+tn)/(tp+tn+fp+fn)
print('accuracy:', acc*100)
print("precission:" ,precission*100)
print("recall:" ,recall*100)
print("f1_score:" ,f1_score*100)