# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:50:46 2016

@author: Joanna
"""
import os

os.chdir ('/home/renato/Desktop/Projects/Trading/')

import numpy as np
from Trading.Training.TradingModel import TradingModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

modelname = 'my_simple_model_v1'
modelpath = './models/weights'
batch_size = 32
lookback_window = 16
n_features = 75

class VerySimpleModel (TradingModel):
    def buildModel (self):
        self.model = Sequential ()
       
        # here, 20-dimensional vectors.
        self.model.add(Dense(256, input_dim=(self.dataset.n_features), init='uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, init='uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, init='uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, init='uniform'))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=custom_objective,
                      optimizer='rmsprop')
    
    def adjustDataset (self):
        self.dataset.X = self.dataset.X[:,-1,:]
        self.dataset.cv_X = self.dataset.cv_X[:,-1,:]
        self.dataset.test_X = self.dataset.test_X[:,-1,:]

    def loadSeriesByNo (self, series_no, bLoadOnlyTrainset=False):
        self.dataset.loadSeriesByNo(series_no, bLoadOnlyTrainset)
        self.adjustDataset ()

    def train (self, verbose=False):
        if len (self.dataset.X.shape) == 3:
            self.dataset.X = self.dataset.X[:,-1,:]

        hist = self.model.fit(self.dataset.X[:-np.mod(len(self.dataset.X),self.batch_size),:],
                            self.dataset.y[:-np.mod(len(self.dataset.X),self.batch_size)],
                            batch_size=self.batch_size, nb_epoch=1, shuffle=False)
    def compute_return (self,ds_sel='cv'):
        if len (self.dataset.X.shape) == 3:
            self.dataset.X = self.dataset.X[:,-1,:]
        if len (self.dataset.cv_X.shape) == 3:
            self.dataset.cv_X = self.dataset.cv_X[:,-1,:]
        if len (self.dataset.test_X.shape) == 3:
            self.dataset.test_X = self.dataset.test_X[:,-1,:]
        self.pred_threshold = 0.33

        return TradingModel.compute_return (self, ds_sel=ds_sel)

if False:        
    my_train = VerySimpleModel (modelname=modelname,
                            modelpath=modelpath)
    
    try:
        #pass
        my_train.loadModel ()
        print ("Features loaded successfully")
        #model.load_weights (model_path+'/'+model_name)
    except:
        pass
    test_and_cv_samples = 2000
    total_no_series = 116
    training_no_series= 15
    
    iterations = 1
    
    my_shape = (iterations,training_no_series)
    
    ret_array = np.zeros(my_shape)
    long_hits_array = np.zeros (my_shape)
    long_misses_array = np.zeros (my_shape)
    short_hits_array = np.zeros (my_shape)
    short_misses_array = np.zeros (my_shape)
    loss_array = np.zeros (my_shape)
    val_loss_array = np.zeros (my_shape)
    
    rdn_ret_array = np.zeros(training_no_series)
    rdn_long_hits_array = np.zeros (training_no_series)
    rdn_long_misses_array = np.zeros (training_no_series)
    rdn_short_hits_array = np.zeros (training_no_series)
    rdn_short_misses_array = np.zeros (training_no_series)
    rdn_loss_array = np.zeros (training_no_series)
    rdn_val_loss_array = np.zeros (training_no_series)
    
    series_list = (np.linspace(1,total_no_series,total_no_series)).astype(int)
    #np.random.shuffle(series_list)
    
    print ("Entering main loop")
    
    if False:
        my_train.loadDataSet(series_list,1,10)
        my_train.trainOnLoadedDataset()
        my_train.evaluateOnLoadedDataset()
        
    
    if False:
        for k in range (iterations):
            
            for counter, series_no in enumerate(series_list[16:training_no_series]):            
                try:
                #if True:
                    if counter == k:
                        my_train.isTraining = True
                    else:
                        my_train.isTraining = False
                    print (series_no)
                    
                    X, y, cv_X, cv_y, test_X, test_y = my_train.loadSeriesByNo (series_no)       
                    
                
                    #checkpoint = ModelCheckpoint(model, monitor='val_acc', save_best_only=True, mode='max')
                    #callbacks_list = [checkpoint]
                
                    print ("Training will start")
                    # Fit the model
                    #hist = model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=callbacks_list,
                    #     validation_data=(test_X, test_y))
                
                    #trains model on one epoch
                    print ("Series no: "+ str(series_no))
                    
                    if my_train.isTraining == True:
                        my_train.train (verbose=True)
                    ret_array[k,counter], long_hits_array[k,counter], long_misses_array[k,counter], short_hits_array[k,counter], short_misses_array[k,counter] = compute_return(my_train.model,cv_X[:-np.mod(len(cv_X),batch_size),:,:], cv_y[:-np.mod(len(cv_X),batch_size)])
                    if k == 0:
                        rdn_ret_array[counter], rdn_long_hits_array[counter], rdn_long_misses_array[counter], rdn_short_hits_array[counter], rdn_short_misses_array[counter] = compute_return(my_rand.model,cv_X[:-np.mod(len(cv_X),batch_size),:,:], cv_y[:-np.mod(len(cv_X),batch_size)])
                    #loss_array[counter] = hist.history['loss'][-1]
                    #val_loss_array[counter] = hist.history['val_loss'][-1]
                    #ret_array[series_no] = ret
                    print ("Return: "+str(ret_array[k, counter])+", "+str(rdn_ret_array[counter]))
                    print ("Long Hits: "+str(long_hits_array[k, counter])+", "+str(rdn_long_hits_array[counter]))
                    print ("Long Misses: "+str(long_misses_array[k, counter])+", "+str(rdn_long_misses_array[counter]))
                    print ("Short Hits: "+str(short_hits_array[k, counter])+", "+str(rdn_short_hits_array[counter]))
                    print ("SHort Misses: "+str(short_misses_array[k, counter])+", "+str(rdn_short_misses_array[counter]))
                    
                    if isTraining == True:
                        my_train.model.save_weights(model_path+'/'+my_train.modelname+".hdf5",overwrite=True)
                    
                    my_train.model.reset_states()
                    #counter += 1
                except:
                    print ("Error - Series: "+ str(series_no))
                    pass
            fig = plt.figure ()
            plt.plot (rdn_ret_array, label="random")
            plt.plot(ret_array[k,:], label="model - "+str(k))
            plt.show ()