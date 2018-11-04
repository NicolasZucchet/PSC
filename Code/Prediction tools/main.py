#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from solutions.CNN import CNN
from solutions.SGDC import SGDC
from solutions.SVC import SVC
import time


##### CNN testing
if True:
    Dataset = ["pamap2"]
    N = [2000]
    Epochs=[500]
    Window_size = [128]
    ### XXX: For harus other window sizes than 128 are not available yet
    
    file = open("logPamap2.txt","a")
    for dataset in Dataset:
        for window_size in Window_size:
            for n in N:
                for epochs in Epochs:
                    t=time.clock()               
                    print("Parameters :\n - dataset = {}\n - n = {}\n - epochs = {}".format(dataset,n,epochs))
                    c = CNN(dataset=dataset,n=n,epochs=epochs,window_size=window_size,heartrate_only=False)
                    c.confusion_matrix()
                    file.write("Parameters :\n - window_size = {}\n - n = {}\n - epochs = {}\n - temps = {}\n".format(window_size,n,epochs,time.clock()-t))
                    file.write(c.accuracy())
                    file.write("\n")
                    file.write(c.top2accuracy())
                    file.write("\n\n")
                    #file.write(c.classification_report())
                    #c.confusion_matrix(normalize=False)
    file.close()
if False:
    Dataset = ["harus"]
    N = [2000]
    Epochs=[500]
    Window_size = [128]
    ### XXX: For harus other window sizes than 128 are not available yet
    
    file = open("logHarus.txt","a")
    for dataset in Dataset:
        for window_size in Window_size:
            for n in N:
                for epochs in Epochs:
                    t=time.clock()               
                    print("Parameters :\n - dataset = {}\n - n = {}\n - epochs = {}".format(dataset,n,epochs))
                    c = CNN(dataset=dataset,n=n,epochs=epochs,window_size=window_size,heartrate_only=False)
                    c.confusion_matrix()
                    file.write("Parameters :\n - window_size = {}\n - n = {}\n - epochs = {}\n - temps = {}\n".format(window_size,n,epochs,time.clock()-t))
                    file.write(c.accuracy())
                    file.write("\n")
                    file.write(c.top2accuracy())
                    file.write("\n\n")
                    c.classification_report()
                    #file.write(c.classification_report())
                    #c.confusion_matrix(normalize=False)
    file.close()


##### SGD Classfier testing
# To choose the subjects to test on, you should modify DataReader class (__data_extraction)
if False:
    Dataset = ["harus"]
    N = [200]
    Window_size = [16]
    for dataset in Dataset:
        for n in N:
            for window_size in Window_size:
                print("Parameters :\n - dataset = {}\n - n = {}\n - window_size = {}".format(dataset,n,window_size))
                s = SGDC(dataset=dataset,n=n,window_size=window_size)
                print(s.accuracy())
                s.confusion_matrix(normalize=True)
                s.classification_report()
                
##### SVC Classifier 
if False:
    Dataset = ["harus"]
    N = [200]
    Window_size = [16]
    for dataset in Dataset:
        for n in N:
            for window_size in Window_size:
                print("Parameters :\n - dataset = {}\n - n = {}\n - window_size = {}".format(dataset,n,window_size))
                s = SVC(dataset=dataset,n=n,window_size=window_size)
                print(s.accuracy())
                s.confusion_matrix(normalize=True)