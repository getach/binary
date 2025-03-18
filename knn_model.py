# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:18:09 2024

@author: getch
"""

from processing import preprocessing 
from sklearn.metrics import confusion_matrix
import numpy as np
import operator
class knn_model:
    def __init__(self,k, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k 
    #@classmethod
    def euclidean_distance(self, x, y):
       return np.linalg.norm(x - y)
    #@classmethod
    def getNeighbors(self, sample_id, K):
       distances = []
       for x in range(len(self.x_train)):
           if (x != sample_id):
               dist = self.euclidean_distance(self.x_train[sample_id], self.x_train[x])
               distances.append((x, dist))
       distances.sort(key=operator.itemgetter(1))
       neighbors = []
       for n in range(K):
           neighbors.append(distances[n][0])
       return neighbors 
    def get_label(self):
      prediction = []
      for i in range(len(self.x_train)):
           neighbors = self.getNeighbors(i, self.k)
           labels = list(y_train[neighbors])
           most_occuring_value = max(labels, key=labels.count)
           prediction.append( most_occuring_value)
      return prediction 
    def evaluate(self,y_train, pred):
        
        tn, fp, fn, tp = confusion_matrix(y_train,pred).ravel()
        #print((tn, fp, fn, tp))
        precission = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*(recall*precission)/(recall+precission)
        acc = (tp+tn)/(tp+tn+fp+fn)
        print('accuracy:', acc*100)
        print("precission:" ,precission*100)
        print("recall:" ,recall*100)
        print("f1_score:" ,f1_score*100)

data = preprocessing('mammographic_masses.data.txt')
x, y = data.process() 
x_train, y_train,x_test,y_test = data.train_test_split(x, y, split_ratio = 0.2)
#print(x_train)
model = knn_model(10, x_train, y_train) 
pred = model.get_label()
acc = model.evaluate(y_train, pred) 
print(acc)  
'''       
def model_accuracy(pred, y_true):
    count = 0
    for i in range(len(y_true)):
        if pred[i]== y_true[i]:
            count+= 1
    pred_accuracy = (count/len(y_true)) *100  
    return pred_accuracy
'''