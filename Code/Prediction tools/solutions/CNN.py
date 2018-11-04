#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### https://github.com/healthDataScience/deep-learning-HAR/blob/master/HAR-CNN.ipynb


### Imports

import numpy as np
import os
import tensorflow as tf
from solutions.Model import Model
from datareader.DataReader import DataReader
from sklearn.preprocessing import LabelBinarizer

### Variables

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

maxint = int(10e10)

class CNN(Model):
    
    ##### Implementation of Model using CNN algorithm
    ### Arguments
        # heartrate_only
    ### Attributes
        # batch size
        # seq_len         Number of steps
        # learning rate
        # epochs
        # dataset         Dataset, in order to use the correct data extractor
        #n_classes        Number of activities
        #n_channels       Number of different input channels
    ### Methods
    
    
    def __init__(self,dataset,n=maxint,window_size=128,epochs=1000,heartrate_only=False):
    
        self.heartrate_only=heartrate_only
        print("-- Début du fitting --")
        self.data = DataReader(dataset,n=n,window_size=window_size,heartrate_only=heartrate_only)
        self.data.split()
        print("-- Fin du fitting --\n")
        print("-- Début du training --")
        self.train(epochs=epochs)
        print("-- Fin du training --\n")


    def __create_network(self):
        
    ### Create the network that will be used for training
        
    ### 1. Instanciate the tensor flow 
        
        self.graph = tf.Graph()

    ### 2. Construct placeholders
        
        with self.graph.as_default():
            if self.heartrate_only:
                inputs_ = tf.placeholder(tf.float32, [None, self.data.window_size, 1], name = 'inputs')
            else:
                inputs_ = tf.placeholder(tf.float32, [None, self.data.window_size, self.data.nb_channels], name = 'inputs')
                
            labels_ = tf.placeholder(tf.float32, [None, self.data.nb_activity], name = 'labels')
            keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
            learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
            
    ### 3. Construct the network
    
        with self.graph.as_default():
            
            # (batch, 128, 9) --> (batch, 64, 18)
            conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
            
            # (batch, 64, 18) --> (batch, 32, 36)
            conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
            
            # (batch, 32, 36) --> (batch, 16, 72)
            conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
            
            # (batch, 16, 72) --> (batch, 8, 144)
            conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
            
        
    ### 4. Flatten and pass to the classifier
    
        with self.graph.as_default():
            
            # Flatten and add dropout
            flat = tf.reshape(max_pool_4, (-1, 8*144))
                # doesn't reshape the first dimension
            flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
                # with prob keep_prob, outputs the input element scaled up by 1 / keep_prob, 
                # otherwise outputs 0
                # The expected sum is unchanged.
        
            # Predictions
            logits = tf.layers.dense(flat,self.data.nb_activity,name="logits")
            proba = tf.nn.softmax(logits, name="probabilities")
                # dense to make the calculation
                # logits sort of probability
            predictions = tf.argmax(logits,1,name="predictions")
                # returns the label with the highest probability
            
            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_),name="cost")
            optimizer = tf.train.AdamOptimizer(learning_rate_,name="optimizer").minimize(cost)
    
            
            # Accuracy
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
                # boolean tensor that says if every value correspond
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
                # good predicition rate
            
            
    def train(self,batch_size=600,seq_len=128,learning_rate=0.0001,epochs=1000):
        
    ### Train the network with the data given
        # batch_size
        # seq_len is the number of data per window
        # learning_rate
        # epochs
        
    ### 1. If fitting has not been made we are raising an error
    
        if not hasattr(self,"data"):
            raise AttributeError("model has not been fitted yet")
       
        
    ### 2. If the folder that will cointains checkpoints is not created we do it
    
        checkpoint_dir = os.path.join(dir_path,"solutions/checkpoints/cnn")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    
    ### 3. We create the graph that will be used
    
        self.__create_network()


    ### 4. We train the network
        
        validation_acc = []
        validation_loss = []
        
        train_acc = []
        train_loss = []
        
        validation_rate = 0.2
        training_size = int((1-validation_rate) * self.data.X_train.shape[0])
        train_X = self.data.X_train[:training_size,:,:]
        train_y = self.data.y_train[:training_size]
        vld_X = self.data.X_train[training_size:,:,:]
        vld_y = self.data.y_train[training_size:]
        
        with self.graph.as_default():
            saver = tf.train.Saver()
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            # Restauring all the tensors and operations needed
            inputs_ = self.graph.get_tensor_by_name("inputs:0")
            labels_ = self.graph.get_tensor_by_name("labels:0")
            keep_prob_ = self.graph.get_tensor_by_name("keep:0")
            learning_rate_ = self.graph.get_tensor_by_name("learning_rate:0")
            cost = self.graph.get_tensor_by_name("cost:0")
            optimizer = self.graph.get_operation_by_name("optimizer")
            accuracy = self.graph.get_tensor_by_name("accuracy:0")
            
            iteration = 1
           
            # Loop over epochs
            for e in range(epochs):
                if e%10 == 0:
                    print(str(e)+"/"+str(epochs))
                # Loop over batches
                for x,y in CNN.__get_batches(train_X, train_y, batch_size):
                    
                    # Feed dictionary
                    feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
                    
                    # Loss
                    loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
                    train_acc.append(acc)
                    train_loss.append(loss)
                    
                    # Compute validation loss at every 10 iterations
                    if (iteration%10 == 0):                
                        val_acc_ = []
                        val_loss_ = []
                        
                        for x_v, y_v in CNN.__get_batches(vld_X, vld_y, batch_size):
                            # Feed
                            
                            feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                            
                            # Loss
                            loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                            val_acc_.append(acc_v)
                            val_loss_.append(loss_v)
                        
                        # Store
                        validation_acc.append(np.mean(val_acc_))
                        validation_loss.append(np.mean(val_loss_))
                    
                    # Iterate 
                    iteration += 1
            
    ### 5. We save the computation done
    
            saver.save(sess,os.path.join(dir_path,"solutions/checkpoints/cnn/TrainedModel.ckpt"))
   
    
    def prediction(self,X):
    
    ### Prediction of the corresponding labels
        # X is the array containing the data
        
    ### 1. Initialization
    
        # Restore last session
        with self.graph.as_default():
            saver = tf.train.Saver()
      
    ### 2. Apply the model computed
            
        with tf.Session(graph=self.graph) as sess:
            
            # Initializes all the variables from the graph
            sess.run(tf.global_variables_initializer())
            
            # Restauring all the tensors and operations needed
            probabilities = self.graph.get_tensor_by_name("probabilities:0")
            predictions = self.graph.get_tensor_by_name("predictions:0")
            inputs_ = self.graph.get_tensor_by_name("inputs:0")
            keep_prob_ = self.graph.get_tensor_by_name("keep:0")
            
            # Feeding inputs to the graph
            feed = {inputs_ : X, keep_prob_ : 1.0}
            
            # Restore the graph created
            saver.restore(sess,os.path.join(dir_path,"solutions/checkpoints/cnn/TrainedModel.ckpt"))
            
            # Makes the prediction
            pred,prob = sess.run([predictions,probabilities],feed_dict=feed)
            
            
            # Activity labels might not start from 1 or 0 and there could miss some activities so
            # we are one-hot encoding the result with a new encoder and using the previous one
            # to decode it
            new_lb = LabelBinarizer()
            new_lb.fit(pred)
            pred = self.data.reverse_one_hot(new_lb.transform(pred))
            
            
            return pred,prob
    
    
    def __get_batches(X, y, batch_size = 100):
        
    ### All the data is not given to the network directly, it is given by groups
            # of batch_size
            # returns a generator with all the batches
        # X, y are the data to turn into batches
        # batch_size is the size of each batch
        
        n_batches = len(X) // batch_size
        X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

        # Loop over batches and yield
        for b in range(0, len(X), batch_size):
            yield X[b:b+batch_size], y[b:b+batch_size]

    


    

