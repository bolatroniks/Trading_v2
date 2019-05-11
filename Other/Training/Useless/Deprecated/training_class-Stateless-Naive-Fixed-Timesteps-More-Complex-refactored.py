# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:50:46 2016

@author: Joanna
"""
#Here I build on the first LSTM design I implemented
#I suppress the moving averages
#I increase the lookback window to 252
#I build a more complex model

import os
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ModelCheckpoint

import theano
import theano.tensor as T

sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts')
sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts/Miscellaneous')
sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts/Miscellaneous/Theano')
sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts/Miscellaneous/Strategy')
sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts/Miscellaneous/Dataset')
sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts/Training')
sys.path.append('/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Scripts/Feature Extractors')

from dataset_func import *
from strategy_func import *
from my_utils import *
from custom_theano import *

from TradingModel import *

        
model_path = '/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Models/Weights'
chars = [-1,0,1]
batch_size = 128
lookback_window = 252
n_features = 75

model_name = 'trained_model_normalized_feat_stateless_fixed_timesteps_more_complex_512_2x_more_dropout'
my_train = TradingModel (model_name, model_path, lookback_window=lookback_window, batch_size=batch_size)
my_rand = TradingModel ("Random_v1", isTraining=False)
my_train.featpath = '/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Featured/Normalized'
my_train.parsedpath = '/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Parsed'
my_train.labelpath = '/home/renato/Desktop/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Labeled'


try:
    #pass
    my_train.loadModel ()
    print ("Model loaded successfully")
    #model.load_weights (model_path+'/'+model_name)
except:
    pass
test_and_cv_samples = 2000
total_no_series = 116
training_no_series= 15

iterations = 1

#my_shape = (iterations,training_no_series)
#
#ret_array = np.zeros(my_shape)
#long_hits_array = np.zeros (my_shape)
#long_misses_array = np.zeros (my_shape)
#short_hits_array = np.zeros (my_shape)
#short_misses_array = np.zeros (my_shape)
#loss_array = np.zeros (my_shape)
#val_loss_array = np.zeros (my_shape)
#
#rdn_ret_array = np.zeros(training_no_series)
#rdn_long_hits_array = np.zeros (training_no_series)
#rdn_long_misses_array = np.zeros (training_no_series)
#rdn_short_hits_array = np.zeros (training_no_series)
#rdn_short_misses_array = np.zeros (training_no_series)
#rdn_loss_array = np.zeros (training_no_series)
#rdn_val_loss_array = np.zeros (training_no_series)

series_list = (np.linspace(1,total_no_series,total_no_series)).astype(int)
#np.random.shuffle(series_list)

print ("Entering main loop")

if False:
    plotSeriesByNo(17, my_train.labelpath)
    my_train.plotPredictionsByNo (17)

if False:
    my_train.loadDataSet(series_list,80,90)
    my_train.evaluateOnLoadedDataset()
    #my_train.trainOnLoadedDataset()

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