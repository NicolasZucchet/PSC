#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

from solutions.Model import Model
from datareader.DataReader import DataReader

class SVC(Model):
    
    def __init__(self,dataset,n=10000000000,window_size=128):
        print("-- Début du fitting --")
        self.data = DataReader(dataset,n=n,window_size=window_size,flatten=True,onehot=False)
        print(self.data.raw_X)
        print(self.data.raw_X.shape)
        print("-- Fin du fitting --\n")
        print("-- Début du training --")
        self.data.split()
        self.training_repartition()
        print(self.data.X_train.shape)
        self.train()
        print("-- Fin du training --\n")

    
    def train(self):
        self.clf = svm.SVC()
        self.clf.fit(self.data.X_train,self.data.y_train)
        
    def prediction(self,X):
        return self.clf.predict(X)
