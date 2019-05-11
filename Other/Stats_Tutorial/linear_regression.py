#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:43:02 2019

@author: joanna
"""

import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression as lr

from Trading.Dataset.Dataset import Dataset
from matplotlib import pyplot as plt

if False:
    ds = Dataset(ccy_pair = 'USD_JPY', from_time = '2012-03-31 00:00:00', to_time = '2014-08-31 00:00:00', timeframe='D')
    ds.bLoadCandlesOnline = False
    ds.loadCandles ()
    ds.loadFeatures ()
    
    df = ds.df
    df['close_lag1'] = df.Close.shift(1)
    
    df['const'] = np.ones (len(df))
    df['ret'] = df.Close - df.close_lag1
    df['ret_lag1'] = df.ret.shift (1)
    df['const'] = np.ones (len(df))
    df['close_lag1'] -= df.close_lag1.mean ()
    
    #df['ret'] -= df.ret.mean ()
    
    eq0 = "ret ~ const - 1"
    eq1 = "ret ~ ret_lag1 + close_lag1 - 1"
    eq2 = "ret ~ ret_lag1"
    eq3 = "ret ~ close_lag1"
    result = sm.ols(formula=eq3, data=df).fit()
    print (result.summary ())
    
    df.Close.plot ()
    
if True:
    ds = Dataset(ccy_pair = 'USD_JPY', from_time = '2012-03-31 00:00:00', to_time = '2014-08-31 00:00:00', timeframe='M15')
    ds.bLoadCandlesOnline = False
    ds.loadCandles ()
    ds.computeFeatures ()
    ds.loadPCAFeatures ()

if False:
    model = lr ()
    model.fit_intercept = True                    
    
    res = model.fit(X=df['ret_lag1', 'close_lag1'], 
                    y = df['ret'])
    
    model = ARIMA(df.ret, order=(2,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())