# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:50:46 2016

@author: Joanna
"""
#Here I build on the first LSTM design I implemented
#I suppress the moving averages
#I increase the lookback window to 252
#I build a more complex model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM

from Trading.Dataset.dataset_func import *
from Trading.Dataset.Dataset import Dataset
from Trading.Training.Useless.Deprecated.Theano.Strategy.strategy_func import *
from Miscellaneous.my_utils import *


class TradingModel ():
    def __init__ (self, modelname, modelpath="./", total_no_series=120, dataset = None,
                  batch_size=128, lookback_window=252, n_features=75, no_signals=3, cv_set_size=1000, test_set_size=1000, pred_threshold=0.7,
                  isTraining=True, isTrainOnlyOnce=False, bZeroMA_train=False, bZeroMA_cv=False, bZeroMA_test=False,
                  featpath = './datasets/Fx/Featured/Normalized_complete',
                  parsedpath = './datasets/Fx/Parsed',
                  labelpath = './datasets/Fx/Labeled'):

        if dataset == None:
            self.dataset = Dataset (lookback_window=lookback_window, n_features=n_features,
                                cv_set_size=cv_set_size, test_set_size = test_set_size,
                                bZeroMA_train = bZeroMA_train,
                                bZeroMA_cv = bZeroMA_cv,
                                bZeroMA_test = bZeroMA_test,
                                featpath = featpath,
                                parsedpath = parsedpath,
                                labelpath = labelpath
                                )
        else:
            self.dataset = dataset

        self.batch_size=128
        
        self.no_signals = no_signals
        self.total_no_series = total_no_series
        self.buildModel ()
        self.modelname = modelname
        self.modelpath = modelpath

        self.pred_threshold = pred_threshold
        self.isTraining = isTraining
        self.isTrainOnlyOnce = isTrainOnlyOnce
    
    def sweepPredThreshold (self, n_pts=10, min_threshold=0.34, max_threshold=0.79, show_plots=False):
        save_threshold = self.pred_threshold
        ret = np.zeros (n_pts)
        l_hits = np.zeros (n_pts)
        l_miss = np.zeros (n_pts)
        s_hits = np.zeros (n_pts)
        s_miss = np.zeros (n_pts)
        ratio = np.zeros (n_pts)
        
        for i in range (n_pts):
            self.pred_threshold = min_threshold + i * (max_threshold - min_threshold) / (n_pts-1)
            print (self.pred_threshold)
            ret[i], l_hits[i], l_miss[i], s_hits[i], s_miss[i] = self.compute_return()
            ratio[i] = (l_hits[i]+s_hits[i])/(l_miss[i]+s_miss[i])
        plt.figure ()
        plt.plot (np.linspace(min_threshold,max_threshold,n_pts), ret)
        plt.show ()
        
        plt.figure ()
        plt.plot (np.linspace(min_threshold,max_threshold,n_pts), ratio)
        plt.show ()
        
        self.pred_threshold = save_threshold
        
        return (ret,l_hits,l_miss, s_hits, s_miss)

    def compute_return (self, ds_sel='cv'):
        return self.dataset.compute_return(self.model, self.pred_threshold,ds_sel)
        
    def loadDataSet (self,series_list=None, begin=0, end=10, bLoadTrainset=True,bLoadCvSet=True,bLoadTestSet=True):
        self.dataset.loadDataSet (series_list, begin, end, bLoadTrainset,bLoadCvSet,bLoadTestSet)
    
    def buildModel (self):
        # build the model: 2 stacked LSTM
        self.model = Sequential()
        self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, stateful=False, return_sequences=False, input_shape=(self.dataset.lookback_window, self.dataset.n_features)))
        self.model.add(Dropout(0.5))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
        #self.model.add(Dropout(0.2))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
        #self.model.add(Dropout(0.5))
        #self.model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
        #self.model.add(Dropout(0.2))
        self.model.add(Dense(512))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(self.no_signals))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        return self.model
    
    def loadModel (self, modelname="",modelpath=""):
        extension = '.hdf5'
        if modelname != "":
            self.modelname = modelname
        if modelpath != "":
            self.modelpath = modelpath    
        self.model.load_weights (self.modelpath+'/'+self.modelname+extension)
            
    def createSingleTrainSet (self):
        self.dataset.createSingleTrainSet()
    
    def trainOnLoadedDataset (self):
        for k in range(len (self.dataset.dataset_list)):
            if np.size(self.dataset.dataset_list[k]) == 6:
                [self.dataset.X, self.dataset.y, self.dataset.cv_X, self.dataset.cv_y, self.dataset.test_X, self.dataset.test_y] = self.dataset.dataset_list[k]
                self.train()
    
    def train (self, shuffle=True, verbose=False):
        if self.isTraining == False:
            return [0]
        if self.isTrainOnlyOnce == True:
            self.isTraining = False
        
        hist = self.model.fit(self.dataset.X[:-np.mod(len(self.dataset.X),self.batch_size),:,:],
                                             self.dataset.y[:-np.mod(len(self.dataset.X),self.batch_size)], 
                                            batch_size=self.batch_size, nb_epoch=1, shuffle=shuffle,
                                            validation_data=(self.dataset.cv_X[:-np.mod(len(self.dataset.cv_X), self.batch_size),:,:], 
                                                                  self.dataset.cv_y[:-np.mod(len(self.dataset.cv_X), self.batch_size)]))
        
        if verbose ==True:
            ret_array, long_hits_array, long_misses_array, short_hits_array, short_misses_array = compute_return(self.model,self.dataset.cv_X[:-np.mod(len(self.dataset.cv_X),batch_size),:,:], self.dataset.cv_y[:-np.mod(len(self.dataset.cv_X),batch_size)])
            
            print ("Return: "+str(ret_array))
            print ("Long Hits: "+str(long_hits_array))
            print ("Long Misses: "+str(long_misses_array))
            print ("Short Hits: "+str(short_hits_array))
            print ("SHort Misses: "+str(short_misses_array))
        return hist
    
    def loadSeriesByNo (self, series_no, bNormalize=True, bLoadOnlyTrainset=False):
        self.dataset.loadSeriesByNo(series_no, bNormalize=bNormalize, bLoadOnlyTrainset=bLoadOnlyTrainset)
        

    def evaluateOnLoadedDataset(self, show_plots=True):
        self.dataset.evaluateOnLoadedDataset(model=self.model, show_plots=show_plots)

    # ToDo: needs refactoring
    def plotPredictionsByNo (self, series_no, data='cv'):
        self.loadSeriesByNo(series_no)
        if data=='cv':
            my_pred = self.model.predict(self.dataset.cv_X)
        elif data=='test':
            my_pred = self.model.predict(self.dataset.test_X)
        fig = plt.figure ()
        plt.plot(my_pred[:,0], label="Short")
        plt.plot(my_pred[:,1], label="Neutral")
        plt.plot(my_pred[:,2], label="Long")
        plt.legend(loc='best')
        plt.show ()
        
    def loadSeriesSequencesIntoRows (self, series_no=1, normalize=True):
        
        self.loadSeriesByNo(series_no)
        X = self.dataset.X
        y = np.zeros((len(X),60, 1))
        
        for i in range (np.size(y,1)):
            y[:,i,:] = relabelDataset(X, period_ahead=i+1)
        
        if normalize == True:
            normalizeOnTheFly(X)
        
        self.X_row = np.zeros ((len(X),np.size(X,1)*np.size(X,2)+np.size(y,1)))
        
        for k in range (len(X)):
            for j in range (np.size(X,1)):
                self.X_row[k, j*np.size(X,2):(j+1)*np.size(X,2)] = X[k,j,:]
                self.X_row[k, -60:] = y[k,:,0]
        #return X_row
    
def appendSequenceRowsToFile (X_row, filename='Master_Normalized.csv', 
                              path='./datasets/Fx/Featured/NotNormalizedNoVolume'):
    
    
    for i in range(len (X_row)):
        with open(path+'/'+filename, 'a') as f:
            f.write (str(X_row[i,0]))
            for j in range (1, np.size(X_row,1)):
                f.write (', '+str(X_row[i,:]))
            f.write ('\n')
            f.close ()