#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:35:23 2018

@author: joanna
"""

from hmmlearn import hmm

from copy import deepcopy
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator


import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class MyHMM ():
    def __init__ (self,
                  no_states = 2,
                  year_start_training = None,
                  training_period = 30,
                  xv_period = 30,
                  pred_delay = 1,
                  bRandomTest = False,
                  prediction_threshold_margin = 0.2,
                  n_iter = 500,
                  cov_type = 'full'
                  ):
        self.no_states = no_states
        self.year_start_training = year_start_training
        self.training_period = training_period
        self.xv_period = xv_period
        self.pred_delay = pred_delay
        self.bRandomTest = bRandomTest
        self.prediction_threshold_margin = prediction_threshold_margin
        self.n_iter = n_iter
        self.cov_type = cov_type
        
        self.model = hmm.GaussianHMM(n_components=self.no_states, 
                                covariance_type=self.cov_type, n_iter = self.n_iter)
        
    def fit (self, 
             df, 
             cols, #observables
             col_discriminator = None, #column used to pick states for later predictions
             high_state_label = 'high_state',
             low_state_label = 'low_state',
             ):
        #saves training data for later use
        self.df = deepcopy (df)
        self.cols = deepcopy (cols)
        
        if self.year_start_training is None:
            self.year_start_training = self.df.index[0].year
        
        start_date = pd.tslib.Timestamp(str (self.year_start_training) + '-01-01 00:00:00')
        cutoff_date = pd.tslib.Timestamp(str (self.year_start_training + self.training_period - 1) + '-12-01 00:00:00')
        
        self.df = self.df.loc[start_date:cutoff_date]
        
        self.df.dropna (inplace = True)
        
        X = np.reshape(self.df[self.cols], (len(self.df),len (self.cols)))
        self.model.fit (X)
        
        self.xv_df = deepcopy(df)        
        self.xv_start_date = pd.tslib.Timestamp(str (self.year_start_training + self.training_period) + '-01-01 00:00:00')
        self.xv_cutoff_date = pd.tslib.Timestamp(str (self.year_start_training + self.training_period + self.xv_period) + '-01-01 00:00:00')            
        self.xv_df = self.xv_df.loc[self.xv_start_date:self.xv_cutoff_date]
        
        if col_discriminator is not None:
            #finds out which state corresponds to recession        
            self.high_state = np.argmax ([(self.df[self.model.predict(X) == i][col_discriminator].mean ()) for i in range (self.no_states)])
            self.low_state = np.argmin ([(self.df[self.model.predict(X) == i][col_discriminator].mean ()) for i in range (self.no_states)])
            self.high_state_label = high_state_label
            self.low_state_label = low_state_label
        
    def predict (self, xv_df = None):
        
        if xv_df is not None:
            self.xv_df = deepcopy(xv_df)        
            self.xv_start_date = pd.tslib.Timestamp(str (self.year_start_training + self.training_period) + '-01-01 00:00:00')
            self.xv_cutoff_date = pd.tslib.Timestamp(str (self.year_start_training + self.training_period + self.xv_period) + '-01-01 00:00:00')            
            self.xv_df = self.xv_df.loc[self.xv_start_date:self.xv_cutoff_date]
            
        if self.bRandomTest:                            
            for col in self.cols:
                #mean = self.xv_df[col].mean ()
                std = self.xv_df[col].std ()
                del self.xv_df[col]
                self.xv_df[col] = np.zeros (len (self.xv_df))
                self.xv_df[col] = np.random.randn (len(self.xv_df[col])) * std
        
        data = np.array(self.df[self.cols])
        idx_list = list(deepcopy(self.df.index))
        
        self.xv_df['model_' + self.high_state_label + '_prob'] = np.zeros (len (self.xv_df))
        self.xv_df['model_' + self.low_state_label + '_prob'] = np.zeros (len (self.xv_df))
        
        for idx in self.xv_df.index:
            data = np.vstack ((data, np.array(self.xv_df[self.cols].loc[idx].values)))
            idx_list.append (idx)
            
            X = pd.core.frame.DataFrame (data = data, 
                                         columns = self.cols, 
                                         index = idx_list)
            state_probs = self.model.predict_proba (X)
            
            self.xv_df['model_' + self.high_state_label + '_prob'][idx] = state_probs[:,self.high_state][-1]
            self.xv_df['model_' + self.low_state_label + '_prob'][idx] = state_probs[:,self.low_state][-1]
                     
        return self.xv_df

#refactored UST
if False:
    no_states = 2
    year = 1935
    training_period = 35
    xv_period = 35
    pred_delay = 0
    claims_delay = 3
    bRandomTest = True
    np.random.seed (35)
    prediction_threshold_margin = 0.5
        
    filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/treasury_10y.csv'
    df = pd.read_csv (filename, 
                          parse_dates=['Date'], 
                          index_col='Date', 
                          infer_datetime_format=True).sort(ascending=True).dropna ()
    
    df['Change'] = (df.SPX / df.SPX.shift(1) - 1)
    
    df.dropna (inplace=True)
    
    my_hmm = MyHMM (no_states = no_states,
                    year_start_training = year,
                    training_period = training_period,
                    xv_period = xv_period,
                    pred_delay = pred_delay,
                    bRandomTest = bRandomTest)
    
    my_hmm.fit(df = df, cols = ['Change'], col_discriminator= 'Change', high_state_label = 'widen', low_state_label = 'tighten')  
    my_hmm.predict ()
    
    xv_df = deepcopy (my_hmm.xv_df)    
    
    #risk parity strategy
    bRiskParity = False
    max_pos = 1.0
    xv_df['vol'] = xv_df.Change.rolling (window=6).std ()
    xv_df['Position'] = np.zeros (len(xv_df))
    xv_df['Position'][xv_df[u'model_tighten_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = -1 * np.minimum(np.ones (len (xv_df)) * (0.02 / xv_df['vol'] if bRiskParity else 1.0), max_pos)
    xv_df['Position'][xv_df[u'model_widen_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = 1 * np.minimum(np.ones (len (xv_df)) * (0.02 / xv_df['vol'] if bRiskParity else 1.0), max_pos)
    xv_df['Change_fwd'] = (xv_df.SPX.shift(-1) - xv_df.SPX) * 100.0
    xv_df['PnL'] = np.cumsum(xv_df['Change_fwd'] * xv_df.Position)
    
    fig = plt.figure (figsize=(14,8))
    xv_df.PnL.plot (label='Strategy return')
    (xv_df.SPX/(xv_df.SPX[0])).plot (label='SPX')
    #np.cumprod(1 + spx_df['FEDFUNDS']/100.0/12).plot (label = 'risk free')
    xv_df.Position.plot ()
    plt.legend (loc='best')
    plt.show ()

#refactored claims vs SPX
if False:
    no_states = 3
    year = 1967
    training_period = 30
    xv_period = 10
    pred_delay = 1
    claims_delay = 3
    bRandomTest = True
    np.random.seed (666)
    prediction_threshold_margin = 0.2    
        
    filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/initial_claims.csv'
    df = pd.read_csv (filename, 
                          parse_dates=['DATE'], 
                          index_col='DATE', 
                          infer_datetime_format=True)
    
    feat = 'ICSA'
    df['Change'] = np.log(df[feat].rolling (window = 4).mean () / df[feat].shift (4).rolling (window = 12).mean ())
    
    my_hmm = MyHMM (no_states = no_states,
                    year_start_training = year,
                    training_period = training_period,
                    xv_period = xv_period,
                    pred_delay = pred_delay,
                    bRandomTest = bRandomTest)
    
    my_hmm.fit(df = df, cols = ['Change'], col_discriminator= 'Change', high_state_label = 'recession', low_state_label = 'expansion')    
    my_hmm.predict ()
    
    spx_filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/spx.csv'
    spx_df = pd.read_csv (spx_filename, 
                          parse_dates=['Date'], 
                          index_col='Date', 
                          infer_datetime_format=True).sort(ascending=True).loc[my_hmm.xv_start_date:my_hmm.xv_cutoff_date]
    
    new_df = spx_df.join(my_hmm.xv_df, how='outer').sort ()
    for col in my_hmm.xv_df.columns:
        new_df[col] = new_df[col].fillna (method = 'ffill')
    
    new_df.dropna ()
    
    #loads risk free rate
    eff_filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/FEDFUNDS.csv'
    ff_df = pd.read_csv (eff_filename, 
                          parse_dates=['DATE'], 
                          index_col='DATE', 
                          infer_datetime_format=True).sort(ascending=True)
    
    new_df = new_df.join(ff_df, how='outer').dropna()
    
    new_df['Return'] = (new_df.SPX / new_df.SPX.shift(1) - 1)
    #risk parity strategy
    bRiskParity = True
    max_pos = 1.0
    new_df['vol'] = new_df.Return.rolling (window=6).std ()
    new_df['Position'] = np.zeros (len(new_df))
    new_df['Position'][new_df[u'model_recession_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = -1 * np.minimum(np.ones (len (new_df)) * (0.02 / new_df['vol'] if bRiskParity else 1.0), max_pos)
    new_df['Position'][new_df[u'model_recession_prob'].shift(pred_delay) < (1.0 - prediction_threshold_margin) * (1.0 / no_states)] = 1 * np.minimum(np.ones (len (new_df)) * (0.02 / new_df['vol'] if bRiskParity else 1.0), max_pos)
    new_df['Ret_fwd'] = new_df.SPX.shift(-1) / new_df.SPX - 1.0
    new_df['PnL'] = np.cumprod(1 + new_df['FEDFUNDS']/100.0 * 1.0 / 12 + new_df['Position'] * (new_df['Ret_fwd'])).shift (1)
    
    fig = plt.figure (figsize=(14,8))
    new_df.PnL.plot (label='Strategy return')
    (new_df.SPX/(new_df.SPX[0])).plot (label='SPX')
    np.cumprod(1 + new_df['FEDFUNDS']/100.0/12).plot (label = 'risk free')
    new_df.Position.plot ()
    plt.legend (loc='best')
    plt.show ()

#refactored SPX price only
ret_list = []
#for seed in range (200):
if True:
    no_states = 3
    year = 1960
    training_period = 30
    xv_period = 20
    pred_delay = 1
    claims_delay = 3
    bRandomTest = False
    np.random.seed (seed)
    prediction_threshold_margin = 0.5
        
    spx_filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/spx.csv'
    df = pd.read_csv (spx_filename, 
                          parse_dates=['Date'], 
                          index_col='Date', 
                          infer_datetime_format=True).sort (ascending = True)
    
    
    df['Change'] = (df.SPX / df.SPX.shift(1) - 1)
    
    df.dropna (inplace = True)
    
    my_hmm = MyHMM (no_states = no_states,
                    year_start_training = year,
                    training_period = training_period,
                    xv_period = xv_period,
                    pred_delay = pred_delay,
                    bRandomTest = bRandomTest)
    
    my_hmm.fit(df = df, cols = ['Change'], col_discriminator= 'Change', high_state_label = 'bull', low_state_label = 'bear')  
    my_hmm.predict ()
    
    #loads risk free rate
    eff_filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/FEDFUNDS.csv'
    ff_df = pd.read_csv (eff_filename, 
                          parse_dates=['DATE'], 
                          index_col='DATE', 
                          infer_datetime_format=True).sort(ascending=True)
    
    new_df = my_hmm.xv_df.join(ff_df, how='outer').dropna()
    
    new_df.SPX = np.cumproduct( 1 + new_df.Change)
    
    new_df['Return'] = (new_df.SPX / new_df.SPX.shift(1) - 1)
    #risk parity strategy
    bRiskParity = True
    max_pos = 1.0
    new_df['vol'] = new_df.Return.rolling (window=6).std ()
    new_df['Position'] = np.zeros (len(new_df))
    new_df['Position'][new_df[u'model_bear_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = -1 * np.minimum(np.ones (len (new_df)) * (0.02 / new_df['vol'] if bRiskParity else 1.0), max_pos)
    new_df['Position'][new_df[u'model_bull_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = 1 * np.minimum(np.ones (len (new_df)) * (0.02 / new_df['vol'] if bRiskParity else 1.0), max_pos)
    new_df['Ret_fwd'] = new_df.SPX.shift(-1) / new_df.SPX - 1.0
    new_df['PnL'] = np.cumprod(1 + new_df['FEDFUNDS']/100.0 * 1.0 / 12 + new_df['Position'] * (new_df['Ret_fwd']))
    
    fig = plt.figure (figsize=(14,8))
    new_df.PnL.plot (label='Strategy return')
    (new_df.SPX/(new_df.SPX[0])).plot (label='SPX')
    new_df ['risk_free'] = np.cumprod(1 + new_df['FEDFUNDS']/100.0/12)
    new_df.risk_free.plot (label = 'risk free')
    new_df.Position.plot ()
    plt.legend (loc='best')
    plt.show ()
    
    ret_list.append (new_df.PnL[-2] / new_df.risk_free[-2])