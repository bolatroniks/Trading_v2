#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:35:27 2017

@author: renato
"""

import numpy
import Pycluster
from sklearn.cluster import KMeans
import numpy as np
points = numpy.vstack([numpy.random.multivariate_normal(mean, 
                                                            0.03 * numpy.diag([1,1]),
                                                            20) 
                           for mean in [(1, 1), (2, 4), (3, 2)]])
points.shape
kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
kmeans
kmeans.labels_
kmeans.labels_.shape
plt.plot(kmeans.labels_)
from matplotlib import pyplot as plt
plt.plot(kmeans.labels_)
kmeans.predict([[0, 0], [4, 4]])
kmeans.cluster_centers_
from Trading.Dataset.Dataset import Dataset
ds = Dataset(ccy_pair='USD_ZAR', from_time=2000, to_time=2010)
ds.initOnlineConfig ()
ds.loadCandles ()
ds.loadFeatures ()
ds.f_df.shape
X = np.zeros ((2067, 4))
X[:,0] = (ds.f_df.Close [1,:] / ds.f_df.Close[0:-1] - 1) / ds.f_df.hist_vol_1y_close[0,:]
X[:,0] = (ds.f_df.Close [1,:] / ds.f_df.Close[0:-1] - 1) / ds.f_df.hist_vol_1y_close[0:]
ds.f_df.hist_vol_1y_close[0:].shape
X[:,0] = (ds.f_df.Close [1,:] / ds.f_df.Close[0:-1] - 1) / ds.f_df.hist_vol_1y_close[0:-1]
X[:,0] = (ds.f_df.Close [1:] / ds.f_df.Close[0:-1] - 1) / ds.f_df.hist_vol_1y_close[0:-1]
ds.f_df.Close [1:].shape
ds.f_df.Close[0:-1].shape
ds.f_df.hist_vol_1y_close[0:-1].shape
X[:,0].shape
X[:,0] = (ds.f_df.Close [1:] / ds.f_df.Close[0:-1] - 1) / ds.f_df.hist_vol_1y_close[0:-1]
X[:,0] = (ds.f_df.Close [1:] / ds.f_df.Close[0:-1] - 1)
X[:,0] = (ds.f_df.Close [1:])
X[:,0] = ds.f_df.Close [1:] / ds.f_df.Close[0:-1] - 1
X[:,0] = ds.f_df.Close [1:] / ds.f_df.Close[0:-1]
ds.f_df.Close[0:-1].shape
X[:,0] = ds.f_df.Close [1:]
ds.f_df.Close [1:] / ds.f_df.Close[0:-1]
(ds.f_df.Close [1:] / ds.f_df.Close[0:-1]).shape
ds.f_df.Close [1:].shape
ds.f_df.Close [1:] / ds.f_df.Close [1:]
ds.f_df.Close [1:] / ds.f_df.Close [1:].shape
(ds.f_df.Close [1:] / ds.f_df.Close [1:]).shape
(ds.f_df.Close [1:] / ds.f_df.Close [0:-1]).shape
(ds.f_df.Close [1:] / ds.f_df.Close [0:-2]).shape
(ds.f_df.Close [1:] / ds.f_df.Close [0:-10]).shape
a = ds.f_df.Close [1:]
b = ds.f_df.Close [0:-1]
a.shape
b.shape
(a/b).shape
(np.array(a)/np.array(b)).shape
X[:,0] = np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close) + 1.0
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) + 1.0
plt.plot(X[:,0])
plt.plot(ds.f_df.Close)
plt.plot(ds.f_df.hist_vol_1y_close)
ds.computeFeatures ()
plt.plot(ds.f_df.hist_vol_1y_close)
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) + 1.0
X = np.zeros ((len(ds.f_df), 4))
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) + 1.0
X = np.zeros ((len(ds.f_df-1), 4))
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) + 1.0
X.shape
len(ds.f_df)
X = np.zeros ((len(ds.f_df-1), 4))
X.shape
X = np.zeros ((len(ds.f_df)-1, 4))
X.shape
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) + 1.0
plt.plot(X[:,0])
plt.plot(ds.f_df.hist_vol_1y_close)
plt.plot( (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) )
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) * 0.01 + 1.0
plt.plot(X[:,0])
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) * 0.1 + 1.0
plt.plot(X[:,0])
X[:,0] = (np.array(ds.f_df.Close [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) * 0.1
plt.plot(X[:,0])
X[:,1] = (np.array(ds.f_df.Open [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) * 0.1
X[:,2] = (np.array(ds.f_df.High [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) * 0.1
X[:,3] = (np.array(ds.f_df.Low [1:]) / np.array(ds.f_df.Close[0:-1]) - 1.0) / np.array(ds.f_df.hist_vol_1y_close[0:-1]) * 0.1
kmeans = KMeans(n_clusters=256, random_state=0).fit(X)
plt.hist(kmeans.labels_, bins=256)
a = kmeans.labels_(kmeans.labels_ == 150)
type(kmeans.labels_)
a = kmeans.labels_[kmeans.labels_ == 150]
a.shape
a
idx = kmeans.labels_ == 150
idx
b = X[idx, :]
b
X[kmeans.labels_ == 151, :]
X[kmeans.labels_ == 152, :]
X[kmeans.labels_ == 153, :]
X[kmeans.labels_ == 154, :]
X[kmeans.labels_ == 255, :]