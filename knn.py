#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:54:19 2024
binary classification using k nearst niebhours algorithim
to run the code using terminal use: python knn.py --input 'mammographic_masses.data.txt'   
@author: getasewalemu
"""
import numpy as np
import operator
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import argparse
parse = argparse.ArgumentParser(description='k nears nighbore algorithm expresion.')
parse.add_argument('--input', type=str, help='please insert an input containing .txt')
args = parse.parse_args()
data = args.input
#df = pd.read_csv('mammographic_masses.data.txt', na_values = '?')
df =  pd.read_csv(data, na_values = '?')
df.columns = ['BI_RADS', 'age', 'shape', 'margin', 'density','severity']
df = df.dropna()
df.reset_index(inplace=True)
scaler = StandardScaler()
x = df[['BI_RADS', 'age', 'shape', 'margin', 'density']]
y = df['severity']
x = scaler.fit_transform(x)
def train_testsplit(x, y, test_ratio):
    np.random.seed(6)
    test_size = int((test_ratio)* df.shape[0])
    #test_size = test_ratio*len(x)
    random_ind = np.random.permutation(df.shape[0])
    x_train = np.array(x[random_ind[test_size:]])
    y_train = np.array(y[random_ind[test_size:]])
    x_test = np.array(x[random_ind[:test_size]])
    y_test = np.array(y[random_ind[:test_size]])
    return x_train, y_train,x_test,y_test   
x_train, y_train, x_test, y_test = train_testsplit(x, y, test_ratio = 0.2)
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def getNeighbors(sample_id, K):
    distances = []
    for x in range(len(x_train)):
        if (x != sample_id):
            dist = euclidean_distance(x_train[sample_id], x_train[x])
            distances.append((x, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for n in range(K):
        neighbors.append(distances[n][0])
    return neighbors
k = 16
prediction = []
for i in range(len(x_train)):
    neighbors = getNeighbors(i, k)
    labels = list(y_train[neighbors])
    most_occuring_value = max(labels, key=labels.count)
    prediction.append( most_occuring_value)
    
def model_accuracy(pred, y_true):
    count = 0
    for i in range(len(y_true)):
        if pred[i]== y_true[i]:
            count+= 1
    pred_accuracy = (count/len(y_true)) *100  
    return pred_accuracy

tn, fp, fn, tp = confusion_matrix(y_train,prediction).ravel()
print((tn, fp, fn, tp))
precission = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(recall*precission)/(recall+precission)
acc = (tp+tn)/(tp+tn+fp+fn)
print('accuracy:', acc*100)
print("precission:" ,precission*100)
print("recall:" ,recall*100)
print("f1_score:" ,f1_score*100)



