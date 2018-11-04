#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from solutions.CNN import CNN
from solutions.SGDC import SGDC
from solutions.SVC import SVC
import time


##### CNN testing

if True:
    Dataset = ["harus"]
    N = [2000]
    Epochs=[500]
    Window_size = [128]
    ### XXX: For harus other window sizes than 128 are not available
    
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
    file.close()

