#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:18:23 2016

@author: renato
"""


#--------------------------------------------------------------
from Trading.Training.TradingModel import TradingModel
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2

import theano.tensor as T

from Framework.Miscellaneous.my_utils import *

modelpath = './models/weights'
modelname = 'Pure_LSTM_Autoencoder'

epsilon = 1.0e-9
alpha = 0.01
_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    #out = -K.dot(y_true, y_pred)
    out = -K.mean(K.minimum(y_true * y_pred, 50), axis=-1)
    #out = -K.mean(y_true * y_pred, axis=-1)
    #out2 = K.mean(y_pred**2)
    #out = -K.mean(y_true * y_pred, axis=-1)
    return out #+ out2 # + K.std((y_true * y_pred)*alpha, axis=-1)
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


 
class PredictAheadModel (TradingModel):
    
    def plotPredictionsByNo (self, series_no, data='cv', X=None):
        
        
        if X == None:
            self.loadSeriesByNo(series_no)
            self.dataset.dropCandleStickFeatures()
            if data=='cv':
                X = self.dataset.cv_X
            elif data=='test':
                X = self.dataset.test_X
            elif data==train:
                X = self.dataset.X
        my_pred = self.model.predict (X, batch_size=self.batch_size)
        
        reconstructed = np.zeros (len(X))
        reconstructed = np.cumprod(1+X[:,-1,4])
        
        fig = plt.figure ()
        plt.plot(my_pred, label="Return 7 days ahead")
        
        #plt.plot(my_pred[:,1], label="Neutral")
        #plt.plot(my_pred[:,2], label="Long")
        plt.legend(loc='best')
        plt.show ()
        
        fig=plt.figure ()
        plt.plot(reconstructed)
        plt.legend(loc='best')
        plt.show ()
     
    def buildModel (self):
        # build the model: 2 stacked LSTM
        self.model = Sequential()
        self.model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1, stateful=False, return_sequences=True, input_shape=(self.dataset.lookback_window, self.dataset.n_features)))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64, dropout_W=0.1, dropout_U=0.1, stateful=False, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(32, dropout_W=0.1, dropout_U=0.1, stateful=False, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128, dropout_W=0.1, dropout_U=0.1, stateful=False, return_sequences=False))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(1, W_regularizer=l2(0.00001)))
        #self.model.add(Dense(1, activity_regularizer=activity_l2(0.01)))
        self.model.add(Activation('softsign'))
        #self.model.add(Activation('softmax'))
        self.model.compile(loss=_loss_tensor, 
            optimizer=SGD(lr=0.01))
        
        return self.model
        
my_train = PredictAheadModel (modelname=modelname,
                            modelpath=modelpath, n_features=81, 
                            lookback_window=126, cv_set_size=1500,
                            test_set_size=1000)

#my_train.modelname = 'Pure_LSTM_3x1024_Layers'
try:
    pass
    #my_train.loadModel ()    
except:
    pass

#my_train.dataset.n_features = 142
#my_train.lookback_window = 126
#my_train.batch_size = 128
#my_train.dataset.cv_set_size = 1500
#my_train.dataset.test_set_size = 1000
my_train.buildModel()
l = my_train.model.layers[-2]
w = l.get_weights () #classifier initial weights
w[1][0] = 0

w2 = my_train.model.layers[-4].get_weights ()

try:
    #my_train.modelname = 'From_scratch_1'
    #my_train.loadModel()
    pass
except:
    print ('Failed to load model')
    pass

my_train.dataset.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume'

cv_X = []
cv_y = []

series_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,40,41,42,43,44,46,47,48,49,50,
             51,52,53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 
             72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 98, 99,
             100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]



logpath = './models/performance/'

if False:
    #my_train.modelname = 'From_scratch_0_v4'
    try:
        my_train.loadModel ()
        
    except:
        pass
    for series_no in series_list:
        try:
            my_train.loadSeriesByNo(series_no)
            my_train.dataset.dropCandleStickFeatures()
            f = open (logpath+my_train.modelname,'a')
            
        #    my_train.dataset.y = relabelDataset(my_train.dataset.X)
        #    my_train.dataset.cv_y = relabelDataset(my_train.dataset.cv_X)
        #    my_train.dataset.test_y = relabelDataset(my_train.dataset.test_X)
        #    
        #    my_train.dataset.X = normalizeOnTheFly(my_train.dataset.X)
        #    my_train.dataset.cv_X = normalizeOnTheFly(my_train.dataset.cv_X)
        #    my_train.dataset.test_X = normalizeOnTheFly(my_train.dataset.test_X)
        #    
            pred = my_train.model.predict(my_train.dataset.cv_X, batch_size=128)
            plt.figure()
            plt.plot(pred)
            plt.show ()
            
            #my_train.train ()
            
            #pred = my_train.model.predict(my_train.dataset.cv_X)
            #plt.plot(pred)
            
            ret = np.zeros (len(pred))
            
            for i in range (len(ret)):
                ret[i] = pred[i] * my_train.dataset.cv_y[i]
            plt.plot(ret)
            
            plt.figure()
            plt.plot(np.cumsum(ret))
            plt.show ()
            f.write ('Series: '+str(series_no)+'\n')
            #for k in range (0,len(ret), 10):
            #    f.write (str(np.cumsum(ret[:k]))+', ')
            #f.write ('\n')
            f.write ('Stats: \n')
            f.write ('cum, cum_min, cum_max, avg, std, min, max\n')
            f.write (str(np.cumsum(ret)[-1])+', '+
                     str(np.min(np.cumsum(ret)))+', '+
                     str(np.max(np.cumsum(ret)))+', '+
                     str(np.mean (ret))+', '+
                     str(np.std(ret))+', '+
                     str(np.min(ret))+', '+
                     str(np.max(ret))+'\n')
            f.close ()
        except:
            print ('Error')
            pass
if False:
    fig = plt.figure ()
    plt.plot(pred)
    plt.plot(my_train.dataset.cv_y)
    plt.show ()

if False:
    #my_train.modelname = 'From_scratch_0_v4'
    try:
        my_train.loadModel ()    
    except:
        pass
    my_train.loadSeriesByNo(1)
    my_train.dataset.dropCandleStickFeatures ()
    
    
#    my_train.dataset.y = relabelDataset(my_train.dataset.X)
#    my_train.dataset.cv_y = relabelDataset(my_train.dataset.cv_X)
#    my_train.dataset.test_y = relabelDataset(my_train.dataset.test_X)
#    
#    my_train.dataset.X = normalizeOnTheFly(my_train.dataset.X)
#    my_train.dataset.cv_X = normalizeOnTheFly(my_train.dataset.cv_X)
#    my_train.dataset.test_X = normalizeOnTheFly(my_train.dataset.test_X)
#    
    pred = my_train.model.predict(my_train.dataset.cv_X)
    plt.figure()
    plt.plot(pred)
    plt.show ()
    
    #my_train.train ()
    
    #pred = my_train.model.predict(my_train.dataset.cv_X)
    #plt.plot(pred)
    
    ret = np.zeros (len(pred))
    
    for i in range (len(ret)):
        ret[i] = pred[i] * my_train.dataset.cv_y[i]
    plt.plot(ret)
    
    plt.figure()
    plt.plot(np.cumsum(ret))
    plt.show ()

if False:
    fig = plt.figure ()
    plt.plot(pred)
    plt.plot(my_train.dataset.cv_y)
    plt.show ()


if False:
    my_train.loadSeriesByNo(1)
    my_train.dataset.dropCandleStickFeatures ()
    my_train.train ()
    
for i in range (0):
    my_train.modelname = 'Pure_LSTM_Autoencoder'
    f = open (logpath+my_train.modelname,'a')
    random.shuffle(series_list)
    try:
        #for i in range(40,41,1):
        #for i in [17,40]:
        
        try:
            
            #17 - USDZAr
            #40 - USDBRL
            #41 - EURZAR?
            
            if True:
                try:                
                    #my_train.loadSeriesByNo(series_list[np.int(np.random.uniform(low=0, high=len(series_list)-1))])
                    k = np.mod(i,len(series_list))
                    my_train.loadDataSet(series_list=series_list,begin=k, end=k+12)
                    my_train.createSingleTrainSet()
                    my_train.dataset.dataset_list = []

                    my_train.dataset.dropCandleStickFeatures ()

                    #my_train.dataset.X = rebuildVWithVariableLength (my_train.dataset.X, 90)
                    
                    f.write ('Loaded series:'+str(series_list[k])+', '+str(series_list[k+1])+', '+str(series_list[k+2])+', '+str(series_list[k+3])+', '+str(series_list[k+4])+', '+str(series_list[k+5])+', '+str(series_list[k+6])+', '+str(series_list[k+7])+'\n')
                except:
                    f.write ('Did not manage to load series\n')
                    raise ValueError('Did not manage to load series')
                #my_train.modelname = 'ZAR-late-sparsity'
                #my_train.dataset.loadDataSet(begin=i, end=i+5)
                #my_train.createSingleTrainSet()
                #my_train.loadSeriesByNo(i)
                
                
                #my_train.dataset.cv_X = my_train.dataset.test_X [0:1500,:,:]
                #my_train.dataset.cv_y = my_train.dataset.test_y [0:1500,:]
                
                #if i==0:
                #    cv_X = my_train.dataset.cv_X
                #    cv_y = my_train.dataset.cv_y
                #else:
                #    my_train.dataset.cv_X = cv_X
                #    my_train.dataset.cv_y = cv_y
                
                my_train.model.layers [-2].set_weights (w)
                my_train.model.layers [-4].set_weights (w2)
                
                for j in range (5):
                        
                        
                        #my_train.dataset.X = my_train.dataset.X[0:2800,:,:]
                        #my_train.dataset.y = my_train.dataset.y[0:2800,:]
                        
                        
                        hist = my_train.train(shuffle=True)
                        f.write ('hist - loss:'+str(hist.history['loss'][0])+'val loss: '+str(hist.history['val_loss'][0])+'\n')
                        if np.isnan(hist.history['loss'][0]):
                            my_train.loadModel ()
                            raise ValueError ('Something went wrong during training')
                
                my_train.model.save_weights(modelpath+'/'+my_train.modelname+".hdf5",overwrite=True)
                f.write ('Model updated\n')
                pred_train = my_train.model.predict(my_train.dataset.X[-3000:,:,:])
                pred_cv = my_train.model.predict(my_train.dataset.cv_X)
                
                ret_train = np.zeros (len(pred_train))
                ret_train = pred_train * my_train.dataset.y [-3000:,:]
                ret_cv = np.zeros (len(pred_cv))
                ret_cv = pred_cv * my_train.dataset.cv_y
                
                plt.figure()
                plt.plot(pred_train, label='pred train: ')
                plt.legend()
                #plt.plot(pred_cv)
                plt.show ()
                
                f.write ('Trainset mean: '+str(np.mean (my_train.dataset.y))+'\n')
                f.write ('Pred_train.mean: '+str(np.mean(pred_train))+'\n')
                f.write ('Pred_train.std: '+str(np.std(pred_train))+'\n')
                f.write ('Pred_train.cumret:'+str(np.cumsum(ret_train))+'\n')
                
                f.write ('CV set mean: '+str(np.mean (my_train.dataset.cv_y))+'\n')
                f.write ('Pred_cv.mean: '+str(np.mean(pred_cv))+'\n')
                f.write ('Pred_cv.std: '+str(np.std(pred_cv))+'\n')
                f.write ('Pred_cv.cumret:'+str(np.cumsum(ret_cv))+'\n')
                
                plt.figure()
                plt.plot(pred_cv, label='pred cv: ')
                plt.legend()
                #plt.plot(pred_cv)
                plt.show ()
                
                
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
                
                f.close ()
                
        except:
            print('Error')            
            f.close ()
                
    except:
        print('Error')
        f.close ()
        pass