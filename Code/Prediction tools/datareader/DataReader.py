#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### Imports

import pandas as pd 
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer


### Variables 

maxint = 100000000
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))





########################
### DATAREADER CLASS ###
########################


# The goal of this class is to take raw data and put it into
# a frame that will be common to all the machine learning models




class DataReader:
    
    
    
    ##### Reads the data from the differents data set
    
    ### Arguments 
        # Dataset name
        
    ### Attributes
        # dataset_size   Number of data in the dataset
        # nb_channels    Number of input channels
        # nb_activity    Number of activities
        # windows_size   Number of elements in a window
        # data_path      Where to find data
        # raw_X          Raw input
        # raw_y          Raw output
        # X_test         Testing input
        # y_test         Testing output
        # X_train        Training input
        # y_train        Training output
        # lb             LabelBinarizer use
        # onehot         Boolean, if we want to one-hot encode the data
        
    ### Functions :
        # Split data in training set / testing set
        # Shuffle the data
        # Standardize the input
        # Take a part of the data
        # One-hot encoding : transformation and reverse transformation
    
    
    
    def __init__(self,dataset,n=maxint,random=True,window_size=128,flatten=False,onehot=True,heartrate_only=False):
        
        # Dataset is the name of the dataset from which we have to extract the data
        # onehot if we want to one-hot encode the labels
        
        mapping_dataset = {
                "harus" : 9,
                "pamap2" : 7}
            # Given the data set, gives the nb_activity corresponding
        self.nb_activity = mapping_dataset[dataset]
        self.onehot = onehot
        self.heartrate_only = heartrate_only
        
        if not (dataset in mapping_dataset.keys()):
            raise ValueError("The dataset name given does not exist")
            # If the dataset given does not exist, we raise an exception
            
        self.nb_channels = mapping_dataset[dataset]
        self.window_size = window_size
        self.flatten = flatten
        self.n = n
        
        self.raw_X,self.raw_y = self.__data_extraction(dataset,flatten)
        
        if onehot:
            self.one_hot()
        
        # taking the n first windows
        n = min(maxint,n)
        if not self.flatten:
            self.raw_X = self.raw_X[:n,:,:]
        else:
            self.raw_X = self.raw_X[:n,:]
        self.raw_y = self.raw_y[:n]
        self.dataset_size = len(self.raw_y)
    
        # if random is activated, we shuffle the data
        if random:
            self.__shuffle()
   
    
    # attention à ne pas prendre des activités qui viennent de deux personnes diff
    def __data_extraction(self,dataset,flatten):
    
    ### Finds the directory where the data is and the extract it
        # dataset is the name of the dataset
        
    ### 1. Define the path where to find the data
        str_path = "data/" + dataset
        data_path = os.path.join(dir_path,str_path)
        
    ### 2. For HARUS dataset
    
        if dataset == "harus":
    
            # If we need to extract the name of the labels
            """ 
            label_path = os.path.join(data_path, "activity_labels.txt")
            labels = pd.read_csv(label_path, header = None) """
            
            # Exctraction of the output
            output_path = os.path.join(data_path, "output/output.txt")
            y = pd.read_csv(output_path, header = None)
            y = np.array(y)
            
            # Exctraction of the input regarding the dataset
            input_path = os.path.join(data_path, "input")
            channel_files = os.listdir(input_path)
            channel_files.sort()
            list_of_channels = []
            window_size = 128
            X_aux = np.zeros((len(y), window_size, self.nb_channels))
            i_ch = 0
            
            for fil_ch in channel_files:
                channel_name = fil_ch[:-4]
                list_of_channels.append(channel_name)
                dat_ = pd.read_csv(os.path.join(input_path,fil_ch), delim_whitespace = True, header = None)
                X_aux[:,:,i_ch] = dat_.as_matrix()
                i_ch += 1
                
            # Removing the division into windows of size 128
            new_X = np.zeros((len(y)*window_size,self.nb_channels))
            new_y = np.zeros(len(y)*window_size)
            for i in range(len(y)):
                new_X[i*window_size:(i+1)*window_size] = X_aux[i]
                new_y[i*window_size:(i+1)*window_size] = new_y[i]
                
            # Take only the data we want
            X_aux = new_X[:2*self.n*self.window_size]
            y_aux = new_y[:2*self.n*self.window_size]
            
            # Divide data into windows of the wanted size
            X_aux,y_aux = self.windowing(X_aux,y_aux)
            
            
                
    
    ### 3. For PAMAP2 dataset    
    
        if dataset == "pamap2":
            
            # For each dat files, extract and make the window division of all activities
                # != 0 (moving from an activity to another)              
            
            raw_data_path = os.path.join(data_path,"data")
                
            # Indexes that are interesting for us
            i_timestamp = 0
            i_label = 1
            i_heart_rate = 2
            i_acc = [4,5,6] # Two accelerometers, we only take one
            i_gyro = [10,11,12]
                # there is way more data that can be collected
            if self.heartrate_only:
                i_total = [i_label,i_heart_rate]
            else:
                i_total = [i_label,i_heart_rate] + i_acc + i_gyro
                      
            # Initialisation of the data
            y = []
            
            if self.heartrate_only:
                X_aux = np.empty(shape=[0,self.window_size,1])
            else:
                X_aux = np.empty(shape=[0,self.window_size,self.nb_channels])    
            
            # Data extraction and preprocessing
            to_process_files = os.listdir(raw_data_path)
            to_process_files = to_process_files[:2]
            for file in to_process_files:
                if file[0] != ".":
                    # extraction
                    file_content = pd.read_csv(os.path.join(raw_data_path,file),sep=";")
                    matrix_content = file_content.as_matrix()
                    # takes only the information we need
                    matrix_content = matrix_content[:,i_total]
                    # sperating lables from input
                    label = matrix_content[:,0]
                    inp = matrix_content[:,1:]
                    # removes extra data
                    inp = inp[:self.n*(self.window_size+1)]
                    label = label[:self.n*(self.window_size+1)]
                    # makes the window division
                    inp,label = self.windowing(inp,label)
                    # removing all the activity 0
                    remove_indexes = []
                    for i in range(len(label)):
                        if label[i] == 0:
                            remove_indexes.append(i)
                    label = np.delete(label,remove_indexes)
                    inp = np.delete(inp,remove_indexes,axis=0)
                    # adding the two arrays to global matrix
                    y = np.append(y,label)
                    X_aux = np.append(X_aux,inp,axis=0)  
            
    ### 4. Flatten the matrix if wanted
        if flatten:
            X = np.empty(shape=[len(X_aux),self.window_size*self.nb_channels])
            for i in range(len(X_aux)):
                X[i] = X_aux[i].flatten()
        else:
            X = X_aux
                
    ### 4. Returns both input and output
        return X,y
        
    
    def split(self,number=False,p=0.3,nb_test=1000,nb_train=5000,random=True):
        
    ### Splits the data into training set and testing set
        # number is true when the split is made on numbers
        # number is false when the split is made on pourcentage
        # random is true when the split is made randomly
           
    ### 1. If the split is made on numbers
        
        if number:
            # testing if the split is possible
            if nb_test + nb_train > self.dataset_size:
                raise IndexError("Not enough values to make the wanted partition")
            
            self.X_train = self.raw_X[:nb_train,:,:]
            self.X_test = self.raw_X[nb_train:nb_test+nb_train,:,:]
            
            self.y_train = self.raw_y[:nb_train]
            self.y_test = self.raw_y[nb_train:nb_test+nb_train]
            
    ### 2. If the split is made on pourcentage
        else:
            nb_train = int((1-p) * self.dataset_size)
            
            if not self.flatten:
                self.X_train = self.raw_X[:nb_train,:,:]
                self.X_test = self.raw_X[nb_train:,:,:]
            else:
                self.X_train = self.raw_X[:nb_train,:]
                self.X_test = self.raw_X[nb_train:,:]
            self.y_train = self.raw_y[:nb_train]
            self.y_test = self.raw_y[nb_train:]
            
        
    def __shuffle(self):
        
    ### Shuffeling raw_X and raw_y in order to get different training/testing
            # sets at each iteration
            
    ### 1. Choose a random permutation
    
        random_permutation = np.random.permutation(self.dataset_size)
        
    ### 2. Create of the new arrays
        new_X = np.empty(self.raw_X.shape)
        new_y = np.empty(self.raw_y.shape)
        
    ### 3. Fill the new arrays
        for old_index,new_index in enumerate(random_permutation):
            new_X[new_index] = self.raw_X[old_index]
            new_y[new_index] = self.raw_y[old_index]
            
    ### 4. Put back arrays in attributes
        
        self.raw_X = new_X
        self.raw_y = new_y
        
      
    def standardize(self,b_test=True,b_other=False,other_data=None):
        
    ### Normalization made with mean and std deviation of training set
        # b_test is true if we want to standardize testing data set
        # b_other is true if we want to standardize other data
        # other_data is the data to standardize if b_other 
    
    ### 1. Testing if the split has been made yet
        if not hasattr(self,"X_train"):
            raise AttributeError("splitting has not been made yet")
    
    ### 2. Calculation of the mean and standard deviation of the training set 
        
        self.mean = np.mean(self.X_train, axis=(0,1))[None,None,:]
        self.std_deviation = np.std(self.X_train, axis=(0,1))[None,None,:]
        self.X_train = (self.X_train - self.mean) / self.std_deviation
        
    ### 3. Normalization of testing data
        
        if b_test:
            self.X_test = (self.X_test - self.mean) / self.std_deviation
        
    ### 4. Normalization of other data
        
        if b_other:
            if other_data == None:
                raise NameError("other_data argument in missing")
            else:
                return ((other_data - self.mean) / self.std_deviation)
            

    def one_hot(self):
        
    ### One-hot encode the labels : replace the label array with a binary 
            # matrix (helps to make better prediction) using LabelBinarizer
    
    ### 1. If the creation of LabelBinarizer has not been made yet, we do so
        
        if not hasattr(self,"lb"):
            self.lb = LabelBinarizer()
            self.lb.fit(self.raw_y)
        
    ### 2. Encoding
        
        self.raw_y = self.lb.transform(self.raw_y)
        self.nb_activity = self.raw_y.shape[1]

    def reverse_one_hot(self,array):
        
    ### Reverse the one-hot encoding, using the LabelBinarizer used before
        # array is the binary matrix to reverse
        
        return self.lb.inverse_transform(array)-1
    
    
    def windowing(self,X,y):
        
    ### Turns a [n,x] array into [?,window_size,x], if the same activity is done
        # X and y are the data to be windowed
            # it must be from the same person and y should not be one-hot encoded
        # window_size the size of the window
        
    ### 1. Initializiation
        
        if self.heartrate_only:
            X_window = np.empty(shape=[0,self.window_size,1])
        else:
            X_window = np.empty(shape=[0,self.window_size,self.nb_channels])
        y_window = []

    ### 2. Windowing
        # While there is still data available, we are looking if there is data 
        # that can be group together to form 
        
        i = 0
        iteration = 0
        while i < len(y)-self.window_size:
            
            if iteration % 1000 == 0:
                print(str(i)+"/"+str(len(y)-self.window_size))
           
            # Looking if the window_size next values were the same activity
            
            j = 1
            cont = True
            while (j<self.window_size and cont) :
                cont = (y[i] == y[i+j])
                j += 1
            
            # If all the data in the frame were from the same activity we add the
            # data
            if cont:
                X_window = np.append(X_window,np.array([X[i:i+self.window_size,:]]),axis=0)
                y_window += [y[i]]
                i += self.window_size
            else:
                i += 1
                
            iteration += 1
                
    ### 3. Returns the computed result
        return X_window,y_window
        
    