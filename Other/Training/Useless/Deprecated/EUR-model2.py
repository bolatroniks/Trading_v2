#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:18:23 2016

@author: renato
"""


#--------------------------------------------------------------
from Trading.Training.TradingModel import TradingModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras import backend as K

import theano.tensor as T

from Trading.Dataset.dataset_func import *
from Framework.Miscellaneous.my_utils import *
from Trading.Training.Useless.Deprecated.Theano.custom_theano import *

modelpath = './models/weights'
modelname = 'EUR-from-scratch-3neurons'

epsilon = 1.0e-9
alpha = 0.01
_EPSILON = K.epsilon()

def normalizeOnTheFly (X):
    x_s = np.shape(X)
    
    for i in range (x_s[0]):
        mu = np.mean (X[i,:,0])
        sigma = np.std (X[i,-22:,0])
        #factor = X[i,-1,0]
        #X[i,:,0] = X[i,:,0] / factor
        X[i,:,0] = (X[i,:,0] - mu) / sigma
        X[i,:,1] = (X[i,:,1]  - mu) / sigma
        X[i,:,2] = (X[i,:,2]  - mu) / sigma
        X[i,:,3] = (X[i,:,3]  - mu) / sigma
        X[i,:,5] = (X[i,:,6]  - mu) / sigma
        X[i,:,6] = (X[i,:,7]  - mu) / sigma
        X[i,:,7] = (X[i,:,8]  - mu) / sigma
        X[i,:,8] = (X[i,:,9]  - mu) / sigma
        X[i,:,23] = (X[i,:,23]  - mu) / sigma
        X[i,:,24] = (X[i,:,24]  - mu) / sigma
        X[i,:,25] = (X[i,:,25]  - mu) / sigma
        X[i,:,34] = (X[i,:,34]  - mu) / sigma
        X[i,:,36] = (X[i,:,36]  - mu) / sigma
        X[i,:,46] = (X[i,:,46]  - mu) / sigma
        X[i,:,47] = (X[i,:,47]  - mu) / sigma
        X[i,:,48] = (X[i,:,48]  - mu) / sigma
        X[i,:,49] = (X[i,:,49]  - mu) / sigma
        X[i,:,50] = (X[i,:,50]  - mu) / sigma
        X[i,:,51] = (X[i,:,51]  - mu) / sigma
        X[i,:,56] = (X[i,:,56]  - mu) / sigma
        X[i,:,58] = (X[i,:,58]  - mu) / sigma
        X[i,:,59] = (X[i,:,69]  - mu) / sigma
        X[i,:,72] = (X[i,:,72]  - mu) / sigma
        X[i,:,79] = (X[i,:,79]  - mu) / sigma
        
        X[i,:,16] /= 100
        X[i,:,17] /= 100
        X[i,:,19] /= 100
        X[i,:,20] /= 100
        X[i,:,21] /= 100
        X[i,:,27] /= 100
        X[i,:,28] /= 100
        X[i,:,29] /= 100
        X[i,:,30] /= 100
        X[i,:,31] /= 100
        X[i,:,52] /= 100
        X[i,:,53] /= 100
        X[i,:,54] /= 100
        X[i,:,55] /= 100
        X[i,:,60] /= 100
        X[i,:,61] /= 100
        X[i,:,62] /= 100
        X[i,:,65] /= 100
        X[i,:,66] /= 100
        X[i,:,68] /= 100
        X[i,:,71] /= 100
        X[i,:,73] /= 100
        X[i,:,74] /= 100
        X[i,:,75] /= 100
        X[i,:,76] /= 100
        X[i,:,78] /= 100
        X[i,:,80] /= 100        
        #X[i,:,4] = X[i,:,4] / np.mean (X[i,:,4])
    print ("mu " + str(mu))
    print ("sigma " + str(sigma))

    return (X)

def rebuildVWithVariableLength (input_X,min_length=90):
    max_length = np.shape(input_X)[1]

    X = np.zeros (np.shape(input_X))
    
    for i in range (len(X)):
        pad_length = max_length - np.int(np.random.uniform(low=min_length, high=max_length))
        if pad_length > 0:
            X[i,0:pad_length,:] = 0
            X[i,pad_length:,:] = input_X[i,pad_length:,:]

    return X
    
    


def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    #out = -K.dot(y_true, y_pred)
    out_s = K.mean(y_true [0] * y_pred[0], axis=-1)
    out_l = -K.mean(y_true [2] * y_pred[2], axis=-1)
    #out_sparse = 500 * K.mean (y_pred[0]+y_pred[2])
    #out2 = K.mean(y_pred**2)
    #out = -K.mean(y_true * y_pred, axis=-1)
    return out_s + out_l #+ out_sparse #+ out2 # + K.std((y_true * y_pred)*alpha, axis=-1)
    #return out

def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    #remove all neutral predictions from y_pred and y_true
    #y_true = y_true[y_pred[:,1]!=True]
    #y_pred = y_pred[y_pred[:,1]!=True]    

    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #cce = 
    #cce = T.nnet.kullback_leibler_divergence(y_pred, y_true)
    #cce = -T.dot(y_pred, y_true)
    
    return cce

def relabelDataset (X, period_ahead=None):
    if period_ahead == None:
        period_ahead = np.int(np.random.uniform(low=1, high=22))
        
    y = np.zeros ((len(X),3))
    
    for i in range (len(y) - period_ahead):
        #y[i, 0] = (X[i+period_ahead,-1,0] / X[i,-1,0] - 1)
        #y[i, 0] = ((X[i+period_ahead,-1,0] / X[i,-1,0] - 1) / np.std(X[i,-22:,5]))
        sigma = np.std(X[i,-22:,5])
        y[i, 0] = y[i, 1] = y[i, 2] =(X[i+period_ahead,-1,0] / X[i,-1,0] - 1) #/ sigma   #/(0.08/16)) #* ((252/period_ahead)**0.5)
    return (y)
 
class PredictAheadModel (TradingModel):
    
    def plotPredictionsByNo (self, series_no, data='cv'):
        self.loadSeriesByNo(series_no)
        if data=='cv':
            my_pred = self.model.predict(self.dataset.cv_X)
        elif data=='test':
            my_pred = self.model.predict(self.dataset.test_X)
        fig = plt.figure ()
        plt.plot(my_pred, label="Return 7 days ahead")
        #plt.plot(my_pred[:,1], label="Neutral")
        #plt.plot(my_pred[:,2], label="Long")
        plt.legend(loc='best')
        plt.show ()
     
    def buildModel (self):
        # build the model: 2 stacked LSTM
        self.model = Sequential()
        self.model.add(LSTM(512, dropout_W=0.1, dropout_U=0.1, stateful=False, return_sequences=True, input_shape=(self.dataset.lookback_window, self.dataset.n_features)))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(512, dropout_W=0.1, dropout_U=0.1, return_sequences=False))
        self.model.add(Dropout(0.1))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
        #self.model.add(Dropout(0.2))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(512))
        #self.model.add(Activation('sigmoid'))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(512))
        self.model.add(Dense(512))
        self.model.add(Activation('softsign'))
        
        #self.model.add(Dropout(0.2))
        self.model.add(Dense(512))
        self.model.add(Activation('softsign'))
        #self.model.add(Dropout(0.2))
        
        self.model.add(Dense(3))
        #self.model.add(Dense(1, activity_regularizer=activity_l2(0.01)))
        self.model.add(Activation('softmax'))
        #self.model.add(Activation('softmax'))
        self.model.compile(loss=_loss_tensor, 
            optimizer='adagrad')
        
        return self.model
        
my_train = PredictAheadModel (modelname=modelname,
                            modelpath=modelpath, n_features=142, 
                            lookback_window=126, cv_set_size=1500,
                            test_set_size=1000)
#my_train.dataset.n_features = 142
#my_train.lookback_window = 126
#my_train.batch_size = 128
#my_train.dataset.cv_set_size = 1500
#my_train.dataset.test_set_size = 1000
my_train.buildModel()
try:
    my_train.modelname = 'EUR-from-scratch-3neurons'
    my_train.loadModel()
except:
    print ('Failed to load model')
    pass

my_train.dataset.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume'

cv_X = []
cv_y = []



if False:
    my_train.loadSeriesByNo(1)
    
    
    
    my_train.dataset.y = relabelDataset(my_train.dataset.X)
    my_train.dataset.cv_y = relabelDataset(my_train.dataset.cv_X)
    my_train.dataset.test_y = relabelDataset(my_train.dataset.test_X)
    
    my_train.dataset.X = normalizeOnTheFly(my_train.dataset.X)
    my_train.dataset.cv_X = normalizeOnTheFly(my_train.dataset.cv_X)
    my_train.dataset.test_X = normalizeOnTheFly(my_train.dataset.test_X)
    
    pred = my_train.model.predict(my_train.dataset.cv_X)
    plt.figure()
    plt.plot(pred)
    plt.show ()
    
    #my_train.train ()
    
    #pred = my_train.model.predict(my_train.dataset.cv_X)
    #plt.plot(pred)
    
    ret = np.zeros (len(pred))
    
    #for i in range (len(ret)):
    #    ret[i] = pred[i] * my_train.dataset.cv_y[i,0]
    #plt.plot(ret)
    
    plt.figure()
    plt.plot(np.cumsum(ret))
    plt.show ()

if False:
    fig = plt.figure ()
    plt.plot(pred)
    plt.plot(my_train.dataset.cv_y)
    plt.show ()    

for training in range (1):
    #for i in range(40,41,1):
    #for i in [17,40]:
    for i in [1]:
        #17 - USDZAr
        #40 - USDBRL
        #41 - EURZAR?
        
        if True:
            #my_train.modelname = 'ZAR-late-sparsity'
            #my_train.dataset.loadDataSet(begin=i, end=i+5)
            #my_train.createSingleTrainSet()
            my_train.loadSeriesByNo(i)
            
            plt.figure ()
            plt.plot(my_train.dataset.X[:,-1,0], label='close px')
            plt.show ()
            
            period_ahead = np.int(np.random.uniform(low=5, high=7))
            
            my_train.dataset.y = relabelDataset(my_train.dataset.X, period_ahead=period_ahead)
            my_train.dataset.cv_y = relabelDataset(my_train.dataset.cv_X, period_ahead=period_ahead)
            my_train.dataset.test_y = relabelDataset(my_train.dataset.test_X, period_ahead=period_ahead)
            
            my_train.dataset.X = normalizeOnTheFly(my_train.dataset.X)
            my_train.dataset.cv_X = normalizeOnTheFly(my_train.dataset.cv_X)
            my_train.dataset.test_X = normalizeOnTheFly(my_train.dataset.test_X)
            
            #my_train.dataset.X = rebuildVWithVariableLength(my_train.dataset.X[-1500:,:,:])
            
            my_train.dataset.X = my_train.dataset.X[0:3500,:,:]
            my_train.dataset.y = my_train.dataset.y[0:3500,:]
            #my_train.dataset.cv_X = my_train.dataset.test_X [0:1500,:,:]
            #my_train.dataset.cv_y = my_train.dataset.test_y [0:1500,:]
            
            #if i==0:
            #    cv_X = my_train.dataset.cv_X
            #    cv_y = my_train.dataset.cv_y
            #else:
            #    my_train.dataset.cv_X = cv_X
            #    my_train.dataset.cv_y = cv_y
            
            for j in range (3):
                my_train.train(shuffle=True)
            #my_train.model.save_weights(modelpath+'/'+my_train.modelname+".hdf5",overwrite=True)
            
            pred_train = my_train.model.predict(my_train.dataset.X)
            pred_cv = my_train.model.predict(my_train.dataset.cv_X)
            
            plt.figure()
            plt.plot(pred_train, label='pred train')
            plt.legend()
            #plt.plot(pred_cv)
            plt.show ()
            
            plt.figure()
            plt.plot(pred_cv, label='pred cv')
            plt.legend()
            #plt.plot(pred_cv)
            plt.show ()
            
            ret_train = np.zeros (len(pred_train))
            ret_train = pred_train * my_train.dataset.y
            ret_cv = np.zeros (len(pred_cv))
            ret_cv = pred_cv * my_train.dataset.cv_y
            #for i in range (len(ret)):
            #    if np.abs (pred[i]) > 0.2:
            #    ret[i] = np.max(-0.03, np.min(pred[i] * my_train.dataset.cv_y[i],0.05))
            #plt.plot(ret_train)
            
            plt.figure()
            plt.plot(np.cumsum(ret_cv), label='cv return')
            plt.legend()
            plt.show ()
            
            plt.figure ()
            plt.plot(np.cumsum(ret_train), label='train_return')
            plt.legend()
            plt.show ()

            
        #except:
        #    print('Error')
        #    pass