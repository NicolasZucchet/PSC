#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Imports
import sys
sys.path.append("../data-reader")
from datareader.DataReader import DataReader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools
import numpy as np

### Variables

maxint = 100000000

class Model():
    
    ##### Interface for models
    ### Attributes
        # batch size
        # seq_len         Number of steps
        # learning rate
        # epochs
        # dataset         dataset to use
    
        
    # def train(self):     
    ### train will be defined in instanced versions of the class
    
    # def predict(self):
    ### will be defined in instanced versions of the class
 
    
    def accuracy(self):
        
        prediction, _ = self.prediction(self.data.X_test)
        s = 0
        if self.data.onehot:
            rev = self.data.reverse_one_hot(self.data.y_test)
            for i in range(len(prediction)):
                s += (prediction[i] != rev[i])
        else:
            for i in range(len(prediction)):
                s += (prediction[i] != self.data.y_test[i])
        s = str((len(prediction)-s)/len(prediction))
        return s   
    
    def top2accuracy(self):
        
        _, prob = self.prediction(self.data.X_test)
        print(prob)
        top2 = []
        for tab in prob:
            max1 = tab[0]
            imax1 = 0
            max2 = tab[1]
            imax2 = 1
            for i in range(2,len(tab)):
                if tab[i] > min(max1,max2):
                    if max1 < max2:
                        max1 = tab[i]
                        imax1 = i
                    else:
                        max2 = tab[i]
                        imax2 = i
            if max1 > max2:
                top2 += [[imax1,imax2]]
            else:
                top2 += [[imax2,imax1]]
        print(top2)
        s = 0
        if self.data.onehot:
            rev = self.data.reverse_one_hot(self.data.y_test)
            print(rev)
            se = []
        
            for i in range(len(prob)):
                if not rev[i] in se:
                    se = se + [int(rev[i])]
            se = np.sort(se)
            print(se)
            inv = [0 for i in range(se[-1]+1)]
            for i in range(len(se)):
                inv[se[i]]=i
            for i in range(len(prob)):
                s += (top2[i][0] != inv[int(rev[i])]) * (top2[i][1] != inv[int(rev[i])])
        else:
            for i in range(len(prob)):
                s += (prob[i] != self.data.y_test[i])
        s = str((len(prob)-s)/len(prob))
        print(s)
        return s
    
    def training_repartition(self):
        l = self.data.y_train
        c = np.count_nonzero(l)
        plt.pie(set(l),c)
        plt.plot()
    
    def confusion_matrix(self,normalize=True):
    
    ### Plots a matrix that shows where the mistakes made by the prediction are
    
    ### 1. Computation of the confusion matrix
        prediction,_ = self.prediction(self.data.X_test)
        if self.data.onehot:
            cm = confusion_matrix(self.data.reverse_one_hot(self.data.y_test),prediction)
        else:
            cm = confusion_matrix(self.data.y_test,prediction)
        # Normalization of the matrix
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    ### 2. Plots the matrix
    
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix for " + self.__class__.__name__)
        plt.colorbar()
        #tick_marks = np.unique(prediction)
        #plt.xticks(tick_marks, classes, rotation=45)
        #plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], ".2f"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
    def classification_report(self):
        prediction = self.prediction(self.data.X_test)
        if (self.data.onehot):
            print(classification_report(self.data.reverse_one_hot(self.data.y_test),prediction))
            return print(classification_report(self.data.reverse_one_hot(self.data.y_test),prediction))
        else:
            print(classification_report(self.data.y_test,prediction))