# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:54:14 2019

@author: joanna
"""

# unsupervised greedy layer-wise pretraining for blobs classification problem
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import time
import sys

from Framework.Dataset.Dataset import Dataset
from Config.const_and_paths import *

class NGrams ():    
    def __init__ (self, 
                  n_centers = 128,
                  random_state = 42,
                  model = None,
                  trainX = None,
                  trainy = None,
                  testX = None,
                  testy = None):
        
        if model is not None:
            self.model = model
            self.n_centers = model.n_clusters
        else:
            self.n_centers = n_centers
            self.model = KMeans(n_clusters = self.n_centers, 
                                random_state = random_state)
    
        self.trainX = trainX
        self.trainy = trainy
        
    def fit (self):
        self.model.fit(self.trainX)

N_FEATURES = 4
N_CATEGORIES = N_FEATURES
N_UNITS = 128
SPARSE_REG = 0.001
 

 
# prepare the dataset
def prepare_dataV2(underlying = 'SPX500_USD', timeframe='M15', y_delay = 1,
                   from_time = 2005, to_time = 2013):
	# generate 2d classification dataset
    ds = Dataset (ccy_pair = underlying, timeframe=timeframe, 
                  from_time = from_time,
                  to_time = to_time)
    ds.loadCandles ()
    ds.df['Change'] = np.log(ds.df.Close / ds.df.Close.shift(1))
    ds.df['Vol'] = ds.df.Change.rolling(window = 50).std ()
    
    ds.df.dropna (inplace = True)
    
    #open/high/low/close     
    xopen = [np.log(o/c)/v for o, c, v in zip (ds.df.Open[1:], ds.df.Close[0:-1], ds.df.Vol[0:-1])]
    xclose = [np.log(cl/o)/v for cl, o, v in zip (ds.df.Close[1:], ds.df.Open[1:], ds.df.Vol[0:-1])]
    xhigh = [np.log(h/o)/v for h, o, v in zip (ds.df.High[1:], ds.df.Open[1:], ds.df.Vol[0:-1])]
    xlow = [np.log(l/o)/v for l, o, v in zip (ds.df.Low[1:], ds.df.Close[1:], ds.df.Vol[0:-1])]
    
    #close over open
    cl_o = np.array([np.log(cl/o)/v for cl, o, v in zip (ds.df.Close[1:], ds.df.Open[1:], ds.df.Vol[0:-1])])
    #close over high
    cl_hi = np.array([np.log(cl/hi)/v for cl, hi, v in zip (ds.df.Close[1:], ds.df.High[1:], ds.df.Vol[0:-1])])
    #close_over_low
    cl_lo = np.array([np.log(cl/lo)/v for cl, lo, v in zip (ds.df.Close[1:], ds.df.Low[1:], ds.df.Vol[0:-1])])
    #high_over_low
    hi_lo = np.array([np.log(hi/lo)/v for hi, lo, v in zip (ds.df.High[1:], ds.df.Low[1:], ds.df.Vol[0:-1])])
    
    	#X, y = make_blobs(n_samples=n_samples, centers=N_FEATURES + 1, n_features=N_FEATURES, cluster_std=2, random_state=2)
    	# one hot encode output variable
    X = np.zeros ((len(ds.df) - 1, N_FEATURES))
    X[:,0] = (np.array(xopen) - np.mean (xopen)) * 1.0 / np.std (xopen)
    X[:,1] = np.array(xhigh) * 1.0 / np.std (xhigh)
    X[:,2] = np.array(xlow) * 1.0 / np.std (xlow)
    X[:,3] = (np.array(xclose) - np.mean (xclose)) * 1.0 / np.std (xclose)
    #X[:,4] = (cl_o - cl_o.mean ()) * (1.0 / np.std (cl_o))
    #X[:,5] = cl_hi / cl_hi.std ()
    #X[:,6] = cl_lo / cl_lo.std ()
    #X[:,7] = hi_lo / hi_lo.std ()
    
    y = np.array([np.log(cl_fwd/cl_0)/v for cl_fwd, cl_0, v in zip (ds.df.Close[y_delay:].shift(-y_delay), ds.df.Close[0:-y_delay].shift(-y_delay), ds.df.Vol[0:-y_delay].shift(-y_delay))])
	# split into train and test
    n_train = int(len(X) * 0.99)
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, testX, trainy, testy
 

# prepare data
trainX = None
for underlying in full_instrument_list:
    try:
        a, b, c, d = prepare_dataV2(underlying = underlying, y_delay=10)
        if trainX is None:
            trainX = a
            trainy = c
            testX = b
            testy = d
        elif np.count_nonzero(np.isnan(a)) == 0:
            trainX = np.vstack ((trainX, a))
            print ('TrainX size: ' + str (len (trainX)))
            trainy = np.array (list(trainy) + list(c))
            testX = np.vstack ((testX, b))
            testy = np.array (list(testy) + list(d))
    except:
        pass
#x_mean = trainX.mean (axis = 0)
#x_std = trainX.std (axis = 0)

print ('TrainX size: ' + str (len (trainX)))

cluster_model = KMeans(n_clusters = N_UNITS, random_state = 42)

start = time.time ()
cluster_model.fit(trainX)
print ('Took ' + str (time.time() - start) + 'seconds to fit Kmeans' )

preds = cluster_model.predict(trainX)

freqs = np.array([len(trainX[preds==_]) for _ in range(len (cluster_model.cluster_centers_))])

scores_dict = dict(zip([_ for _ in range(len (cluster_model.cluster_centers_))], 
                   [ (trainy[preds == _].mean () / trainy.std () if freqs[_] > 5 else 0.0) for _ in range(len (cluster_model.cluster_centers_))]))

best_short_idx = np.argmin(list(scores_dict.values ()))
best_long_idx = np.argmax(list(scores_dict.values ()))

print ('Number of centers: ' + str (N_UNITS))
print ('Best long: ' + str (scores_dict[best_long_idx]))
print ('Best short: ' + str (scores_dict[best_short_idx]))

from itertools import product

chars = [_ for _ in range(len (cluster_model.cluster_centers_))]

bigrams = product(chars, chars)
bigrams = [_ for _ in bigrams]
preds_minus_1 = np.zeros (len(preds))
preds_minus_1[1:] = preds[0:-1]
preds_minus_1[0] = 999

bigrams_scores_dict = dict(zip([_ for _ in bigrams], 
                   [ (trainy[(preds == pred_t0) & (preds_minus_1 == pred_t_1)].mean () / trainy[(preds == pred_t0) & (preds_minus_1 == pred_t_1)].std () if len (trainy[(preds == pred_t0) & (preds_minus_1 == pred_t_1)]) >= 2 else 0.0) for (pred_t0, pred_t_1) in bigrams]))
                   
best_short_idx = list(bigrams_scores_dict.keys ())[np.argmin(list(bigrams_scores_dict.values ()))]
best_long_idx = list(bigrams_scores_dict.keys ())[np.argmax(list(bigrams_scores_dict.values ()))]

print ('Number of centers: ' + str (N_UNITS))
print ('Best long: ' + str (bigrams_scores_dict[best_long_idx]))
print ('Best short: ' + str (bigrams_scores_dict[best_short_idx]))

count = 0
for k, v in bigrams_scores_dict.iteritems ():
    if np.abs(v) >= 1.0 and len (trainy[(preds == k[0]) & (preds_minus_1 == k[1])]) > 5:
        print (str(k) + ',' + str(v) + ',' + str (len (trainy[(preds == k[0]) & (preds_minus_1 == k[1])])))
        count += len (trainy[(preds == k[0]) & (preds_minus_1 == k[1])])
#trainX[preds == 30]

avg_error = (np.sum((trainX - cluster_model.cluster_centers_[cluster_model.predict(trainX)]) ** 2) / len (trainX))**0.5
