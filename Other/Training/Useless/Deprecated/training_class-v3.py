# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:50:46 2016

@author: Joanna
"""
#here I introduce a few improvements to the training routine?
    #namely, the traininig set is not shuffled anymore
    #I try to introduce stateful LSTM
    #training on a single batch each time could achieve the same but it would take too long
    #as each gradient descent step would take 1 minute
    #therefore this is impractical for this problem

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
    def __init__ (self, modelname, modelpath="./", featpath="./", parsedpath="./", labelpath="./", isTraining=True, isTrainOnlyOnce=False):
        self.model = self.buildModel ()
        self.modelname = modelname
        self.modelpath = modelpath
        self.featpath = featpath
        self.parsedpath = parsedpath
        self.labelpath = labelpath
        self.isTraining = isTraining
        self.isTrainOnlyOnce = isTrainOnlyOnce
        self.dataset_list = []
        
        self.X = []
        self.y = []
        self.cv_X = []
        self.cv_y = []
        self.test_X = []
        self.test_y = []

    def compute_return (self, dataset='cv'):
        if dataset=='cv':
            X = self.cv_X
            y = self.cv_y
        elif dataset=='train':
            X = self.X
            y = self.y
        elif dataset=='test':
            X = self.test_X
            y = self.test_y
        return compute_return(self.model, X, y)

    def loadDataSet (self,series_list=None, begin=0,end=10):
        self.dataset_list = []
        
        if series_list == None:
            return

        for counter, series_no in enumerate(series_list[begin:end]):
            elem = [my_train.loadSeriesByNo (series_no)]
            self.dataset_list.append (elem)
        return
    
    def buildModel (self, lookback_window=120):
        # build the model: 2 stacked LSTM
        model = Sequential()
        model.add(LSTM(128, stateful=False, input_shape=(lookback_window, n_features)))
        #model.add(Dropout(0.2))
        #model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        model.compile(loss=custom_objective, optimizer='rmsprop')
        
        return model
        #model.compile(loss='kullback_leibler_divergence', optimizer='rmsprop')
    
    def loadModel (self, modelname="",modelpath=""):
        extension = '.hdf5'
        if modelname != "":
            self.modelname = modelname
        if modelpath != "":
            self.modelpath = modelpath    
        self.model.load_weights (self.modelpath+'/'+self.modelname+extension)
    
    def evaluateOnLoadedDataset (self, show_plots=True):
        my_shape = (len(self.dataset_list))

        self.ret_array = np.zeros(my_shape)
        self.long_hits_array = np.zeros (my_shape)
        self.long_misses_array = np.zeros (my_shape)
        self.short_hits_array = np.zeros (my_shape)
        self.short_misses_array = np.zeros (my_shape)
        self.loss_array = np.zeros (my_shape)
        self.val_loss_array = np.zeros (my_shape)
        
        for k in range(len (self.dataset_list)):
            if np.size(self.dataset_list[k],1) == 6:                    
                [self.X, self.y, self.cv_X, self.cv_y, self.test_X, self.test_y] = self.dataset_list[k][0]
                self.ret_array[k], self.long_hits_array[k], self.long_misses_array[k], self.short_hits_array[k], self.short_misses_array[k] = compute_return(self.model,self.cv_X[:-np.mod(len(self.cv_X),batch_size),:,:], self.cv_y[:-np.mod(len(self.cv_X),batch_size)])
        if show_plots == True:
            fig = plt.figure ()
            #for k in range(len (self.dataset_list)):
            plt.plot (self.ret_array, label="Returns")
            plt.legend (loc='best')
            plt.show ()
    
            
    def createSingleTrainSet (self):
       total_len = 0
       lookback_window = np.size(self.dataset_list[0][0][0], 1)
       n_features = np.size(self.dataset_list[0][0][0], 2)
       
       for i in range(len (self.dataset_list)):
           if (np.size(self.dataset_list[i])>0):
               total_len += len(self.dataset_list[i][0][0])
       self.X = np.zeros((total_len, lookback_window, n_features))
       self.y = np.zeros((total_len, 3))
       
       total_len = 0
       for i in range(len (my_train.dataset_list)):
           if (np.size(self.dataset_list[i])>0):
               self.X[total_len:total_len+len(self.dataset_list[i][0][0]),:,:] = self.dataset_list[i][0][0]
               self.y[total_len:total_len+len(self.dataset_list[i][0][0]),:] = self.dataset_list[i][0][1]
               total_len += len(self.dataset_list[i][0][0])
    
    def trainOnLoadedDataset (self):
        for k in range(len (self.dataset_list)):
            if np.size(self.dataset_list[k],1) == 6:                    
                [self.X, self.y, self.cv_X, self.cv_y, self.test_X, self.test_y] = self.dataset_list[k][0]
                self.train()
    
    def train (self, verbose=False):
        if self.isTraining == False:
            return [0]
        if self.isTrainOnlyOnce == True:
            self.isTraining = False
        
        hist = self.model.fit(self.X[:-np.mod(len(self.X),batch_size),:,:], self.y[:-np.mod(len(self.X),batch_size)], batch_size=batch_size, nb_epoch=1, shuffle=False)
                #validation_data=(cv_X[:-np.mod(len(cv_X), batch_size),:,:], cv_y[:-np.mod(len(cv_X), batch_size)]))
        if verbose ==True:
            ret_array, long_hits_array, long_misses_array, short_hits_array, short_misses_array = compute_return(self.model,self.cv_X[:-np.mod(len(self.cv_X),batch_size),:,:], self.cv_y[:-np.mod(len(self.cv_X),batch_size)])
            
            print ("Return: "+str(ret_array))
            print ("Long Hits: "+str(long_hits_array))
            print ("Long Misses: "+str(long_misses_array))
            print ("Short Hits: "+str(short_hits_array))
            print ("SHort Misses: "+str(short_misses_array))
        return hist
        #loss_array[counter] = hist.history['loss'][-1]
        #val_loss_array[counter] = hist.history['val_loss'][-1]
    
    def loadSeriesByNo (self, series_no):
        print ("Loading Series #"+str(series_no))
        
        #---------------load series into dataframe
        parsed_filename = 'ccy_hist_ext_'+str(series_no)+'.txt'
        feat_filename = 'ccy_hist_normalized_feat_'+str(series_no)+'.csv'     
        label_filename = 'ccy_hist_feat_'+str(series_no)+'.csv' #need to fix this name
        try:
            #------------first features
            my_df = loadSeriesToDataframe (self.featpath, feat_filename)
            print ("Length Features Dataframe: "+str(len(my_df)))
            
            #splits dataframe into train, cv and test
            train_df, cv_df, test_df = splitDataframeIntoTrainCVTest (my_df)
            
            #------------then labels        
            my_df = loadSeriesToDataframe (self.labelpath, label_filename)
            print ("Length labels Dataframe: "+str(len(my_df)))
            
            #splits dataframe into training set and cross validation sets
            train_labels_df, cv_labels_df, test_labels_df = splitDataframeIntoTrainCVTest (my_df)
            #---------------------------------------------
            
            #---------buidSentences does not mean anything, need to get rid of this step
            sentences, next_chars = buildSentences (train_df, train_labels_df)
            cv_sentences, cv_next_chars = buildSentences (cv_df, cv_labels_df)
            test_sentences, test_next_chars = buildSentences (test_df, test_labels_df)
        
            #---------initializes train_X, train_y, cv_X and cv_y sets
            self.X, self.y = buildSequencePatches (sentences, next_chars)
            self.cv_X, self.cv_y = buildSequencePatches (cv_sentences, cv_next_chars)
            self.test_X, self.test_y = buildSequencePatches (test_sentences, test_next_chars)
            
            print ("Series #"+str(series_no)+"loaded successfully")
        except:
            print ("Failed to load Series #"+str(series_no))
            return []
        
        return self.X, self.y, self.cv_X, self.cv_y, self.test_X, self.test_y
    
    def plotPredictionsByNo (self, series_no, data='cv'):
        self.loadSeriesByNo(series_no)
        if data=='cv':
            my_pred = self.model.predict(self.cv_X)
        elif data=='test':
            my_pred = self.model.predict(self.test_X)
        fig = plt.figure ()
        plt.plot(my_pred[:,0], label="Short")
        plt.plot(my_pred[:,1], label="Neutral")
        plt.plot(my_pred[:,2], label="Long")
        plt.legend(loc='best')
        plt.show ()
    
def loadSeriesToDataframe (path, filename):
    my_df = pandas.read_csv(path+'/'+filename, converters={'Change':p2f})
    my_df['Date'] = pandas.to_datetime(my_df['Date'],dayfirst=True)
    my_df.index = my_df['Date']
    my_df = my_df.sort(columns='Date', ascending=True)
    del my_df['Date']

    return my_df

def plotSeriesByNo (series_no, path="./", filename_prefix='ccy_hist_feat_', filetype='label', field='Close', df_sel='cv'):
    try:
        my_df = loadSeriesToDataframe (path, filename_prefix+str(series_no)+'.csv')
        train_df, cv_df, test_df = splitDataframeIntoTrainCVTest (my_df)
        if df_sel == 'cv':
            plt.plot(cv_df[field])
        elif df_sel == 'test':
            plt.plot(test_df[field])
    except:
        print ("Error loading dataframe")
    
def splitDataframeIntoTrainCVTest (df, n_cv=1000, n_test=1000):
    #check if df length is enough
    if len(df) < n_cv + n_test:
        print ("Dataframe too small")
    
    train_df = df [0:-n_cv-n_test]
    cv_df = df[-n_test-n_cv:-n_test]
    test_df = df[-n_test:]

    return train_df, cv_df, test_df

def buildSentences (feat_df, labels_df):
    feat_df = feat_df.merge(labels_df)
    feat_df = feat_df.dropna()
    labels_df = feat_df.ix[:,'Labels':]
    del feat_df ['Labels']
 
    sentences = np.zeros ((len(feat_df),len(feat_df.columns)-1))
    next_chars = np.zeros (len(feat_df))    
    #feat_df['ma_200_close'] = 0
    for i in range(len(feat_df)):
        sentences[i,:] = np.array(feat_df.ix[:,'Close':].irow(i).values)
        next_chars [i] = labels_df ['Labels'].irow(i)
    return sentences, next_chars

def buildSequencePatches (sentences, next_chars, lookback_window=120):
    n_features = np.size (sentences, 1)
    X = np.zeros((len(sentences)-lookback_window+1, lookback_window, n_features))
    y = np.zeros((len(sentences)-lookback_window+1,len(chars)), dtype=np.bool)
    #test_X = np.zeros((len(test_sentences)-lookback_window+1, lookback_window, n_features))
    #test_y = np.zeros((len(test_sentences)-lookback_window+1,len(chars)), dtype=np.bool)
    
    #----------builds train_X and train_Y 
    for i in range(len(sentences)-lookback_window+1):
        for j in range (lookback_window):
            X[i,j,:] = sentences[i+j,:]
        
        if next_chars[i] == -1:
            y[i,:] = [1,0,0]
        elif next_chars[i] == 0:
            y[i,:] = [0,1,0]
        elif next_chars[i] == 1:
            y[i,:] = [0,0,1]
    return X, y
        
model_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Models/Weights'
chars = [-1,0,1]
#print('total chars:', len(chars))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
#maxlen = 40
#step = 3
#sentences = []
#next_chars = []
batch_size = 32
lookback_window = 120
n_features = 75
isTraining = True
isTrainOnlyOnce = False

model_name = 'trained_model_normalized_feat_v3'
my_train = My_Training (model_name, model_path, isTrainOnlyOnce=False)
my_rand = My_Training ("Random_v1", isTraining=False)
my_train.featpath = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Featured/Normalized'
my_train.parsedpath = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Parsed'
my_train.labelpath = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Labeled'


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