# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:50:46 2016

@author: Joanna
"""

import numpy as np
import pandas
import time
from time import sleep
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

sys.path.append('C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Scripts')
sys.path.append('C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Scripts/Miscellaneous')
sys.path.append('C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Scripts/Training')
sys.path.append('C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Scripts/Feature Extractors')

from my_utils import *

epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    #remove all neutral predictions from y_pred and y_true
    y_true = y_true[y_pred[:,1]!=True]
    y_pred = y_pred[y_pred[:,1]!=True]    

    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #cce = T.nnet.kullback_leibler_divergence(y_pred, y_true)
    return cce

def compute_return (model, X, y):
    try:
        pred = model.predict_classes(X)
    
        pos_pred = pred[pred!=1]
        pos_y = y[pred!=1]
    
        ret = 0.0
        long_hits = 0
        long_misses = 0
        short_hits = 0
        short_misses = 0
        for j in range(len (pos_pred)):
            if pos_pred [j]== 0:
                if pos_y[j,0] == True:
                    ret += 5
                    short_hits += 1
                else:
                    ret += -3
                    short_misses += 1
            else:
                if pos_y[j,2] == True:
                    ret += 5
                    long_hits += 1
                else:
                    ret += -3
                    long_misses += 1
        return ret, long_hits, long_misses, short_hits, short_misses
    except:
        return 0, 0, 0

class My_Training ():
    def __init__ (self):
        self.model = self.buildModel ()
    
    def buildModel (self):
        
#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
#print('corpus length:', len(text))
model_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Models/Weights'
chars = [-1,0,1]
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
batch_size = 32
lookback_window = 120
n_features = 75

# build the model: 2 stacked LSTM
model = Sequential()
model.add(LSTM(128, stateful=False, input_shape=(lookback_window, n_features)))
#model.add(Dropout(0.2))
#model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss=custom_objective, optimizer='rmsprop')
#model.compile(loss='kullback_leibler_divergence', optimizer='rmsprop')

model_name = 'trained_model_normalized_feat_v1.hdf5'
try:
    pass
    #model.load_weights (model_path+'/'+model_name)
except:
    pass
test_and_cv_samples = 2000
total_no_series = 116
training_no_series= 100

ret_array = np.zeros(training_no_series)
long_hits_array = np.zeros (training_no_series)
long_misses_array = np.zeros (training_no_series)
short_hits_array = np.zeros (training_no_series)
short_misses_array = np.zeros (training_no_series)
loss_array = np.zeros (training_no_series)
val_loss_array = np.zeros (training_no_series)

feat_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Featured/Normalized'
parsed_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Parsed'
label_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Labeled'

series_list = (np.linspace(1,total_no_series,total_no_series)).astype(int)
#np.random.shuffle(series_list)

print ("Entering main loop")
counter = 0
for series_no in series_list[41:training_no_series]:
    try:
    #if True:
        print (series_no)
        
        #---------------load series into dataframe
        parsed_filename = 'ccy_hist_ext_'+str(series_no)+'.txt'
        feat_filename = 'ccy_hist_normalized_feat_'+str(series_no)+'.csv'     
        label_filename = 'ccy_hist_feat_'+str(series_no)+'.csv' #need to fix this name
        
        #------------first features
        my_df = pandas.read_csv(feat_path+'/'+feat_filename, converters={'Change':p2f})
        print ("File read")
        my_df['Date'] = pandas.to_datetime(my_df['Date'],dayfirst=True)
        my_df.index = my_df['Date']
        my_df = my_df.sort(columns='Date', ascending=True)
        del my_df['Date']
        
        #------------then labels
        train_df = my_df[:-test_and_cv_samples]
        test_and_cv_df = my_df[-test_and_cv_samples:]
        
        my_df = pandas.read_csv(label_path+'/'+label_filename, converters={'Change':p2f})
        my_df['Date'] = pandas.to_datetime(my_df['Date'],dayfirst=True)
        my_df.index = my_df['Date']
        my_df = my_df.sort(columns='Date', ascending=True)
        del my_df['Date']

        
        #splits dataframe into training set and cross validation sets
        train_labels_df = my_df[:-test_and_cv_samples]
        test_and_cv_labels_df = my_df[-test_and_cv_samples:]
        #---------------------------------------------
        
        sentences = np.zeros ((len(train_df),len(train_df.columns)-1))
        next_chars = np.zeros (len(train_df))
        test_sentences = np.zeros ((len(test_and_cv_df),len(test_and_cv_df.columns)-1))
        test_next_chars = np.zeros (len(test_and_cv_df))
        
        #---------buids train_X, train_y, cv_X and cv_y sets
        for i in range(len(train_df)):
            sentences[i,:] = np.array(train_df.ix[:,'Close':].irow(i).values)
            next_chars [i] = train_labels_df ['Labels'][i]
    
        for i in range(len(test_and_cv_df)):
            test_sentences[i,:] = np.array(test_and_cv_df.ix[:,'Close':].irow(i).values)
            test_next_chars [i] = test_and_cv_labels_df ['Labels'][i]
        
        #---------initializes train_X, train_y, cv_X and cv_y sets 
        X = np.zeros((len(sentences)-lookback_window+1, lookback_window, n_features))
        y = np.zeros((len(sentences)-lookback_window+1,len(chars)), dtype=np.bool)
        test_X = np.zeros((len(test_sentences)-lookback_window+1, lookback_window, n_features))
        test_y = np.zeros((len(test_sentences)-lookback_window+1,len(chars)), dtype=np.bool)
        
        #----------builds train_X and train_Y 
        for i in range(len(sentences)-lookback_window+1):
            for j in range (lookback_window):
                X[i,j,:] = (sentences[i+j,:] - np.mean(sentences[i+j,:])) / np.std(sentences[i+j,:])
            
            if next_chars[i] == -1:
                y[i,:] = [1,0,0]
            elif next_chars[i] == 0:
                y[i,:] = [0,1,0]
            elif next_chars[i] == 1:
                y[i,:] = [0,0,1]
        #-----------------------------------
        #----------builds cv_X and cv_y        
        for i in range(len(test_sentences)-lookback_window+1):        
            for j in range (lookback_window):
                test_X[i,j,:] = (test_sentences[i+j,:] - np.mean(test_sentences[i+j,:])) / np.std(test_sentences[i+j,:])
            
            #test_X[i,:,0] = test_sentences[i,:]        
            if test_next_chars[i] == -1:
                test_y[i,:] = [1,0,0]
            elif test_next_chars[i] == 0:
                test_y[i,:] = [0,1,0]
            elif test_next_chars[i] == 1:
                test_y[i,:] = [0,0,1]
    
        checkpoint = ModelCheckpoint(model, monitor='val_acc', save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
    
        print ("Training will start")
        # Fit the model
        #hist = model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=callbacks_list,
        #     validation_data=(test_X, test_y))
    
        #trains model on one epoch
        print ("Series no: "+ str(series_no))
        for iteration in range(1, 2):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            hist = model.fit(X[300:-np.mod(len(X)-300,batch_size),:,:], y[300:-np.mod(len(X)-300,batch_size)], batch_size=batch_size, nb_epoch=1,
                      validation_data=(test_X[:-np.mod(len(test_X), batch_size),:,:], test_y[:-np.mod(len(test_X), batch_size)]))
            ret_array[counter], long_hits_array[counter], long_misses_array[counter], short_hits_array[counter], short_misses_array[counter] = compute_return(model,test_X[:-np.mod(len(test_X),batch_size),:,:], test_y[:-np.mod(len(test_X),batch_size)])
            loss_array[counter] = hist.history['loss'][-1]
            val_loss_array[counter] = hist.history['val_loss'][-1]
            #ret_array[series_no] = ret
            print ("Return: "+str(ret_array[counter]))
            print ("Long Hits: "+str(long_hits_array[counter]))
            print ("Long Misses: "+str(long_misses_array[counter]))
            print ("Short Hits: "+str(short_hits_array[counter]))
            print ("SHort Misses: "+str(short_misses_array[counter]))
        model.save_weights(model_path+'/'+model_name,overwrite=True)
        
        model.reset_states()
        counter += 1
    except:
        pass
