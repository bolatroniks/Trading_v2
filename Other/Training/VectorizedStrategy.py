# -*- coding: utf-8 -*-
from Framework.Dataset.DatasetHolder import *

from Framework.Features.TimeSeries import halflife

import pandas as pd
import numpy as np

from hashlib import sha1

from Framework.Strategy.Utils.strategy_func import compute_hit_miss_array, plot_signals, plot_pnl, plot_histogram

PCA_SUFFIX = '012_'


def compute_predictions_simple (ds):
    df = ds.f_df #just a short name
    
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    #df['trendlines_diff_10_D'] = df['no_standing_upward_lines_10_D'] - df['no_standing_downward_lines_10_D']
    #features below look at the difference between current value of the above feature and its max/min
    #df['trend_diff_change_up_D'] = (df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).min())
    #df['trend_diff_change_down_D'] = -(df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).max())

    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI_' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).min ()
    df['max_RSI' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))

    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    preds[(df['RSI_' + fast_timeframe] < 30) & \
          (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe ] > 50)] = 1
   
    #sells if:
    #opposite of above
    preds[(df['RSI_' + fast_timeframe] > 70) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50)] = -1    
    

    return preds


#This is the core of the strategy, 
#the bit that generates signals based on the features up to time t

def compute_predictions (ds, **kwargs):
    print (str(kwargs))
    
    if 'fast_timeframe' in kwargs.keys ():
        fast_timeframe = kwargs['fast_timeframe']
    else:
        fast_timeframe = 'M15'
        
    if 'slow_timeframe' in kwargs.keys ():
        slow_timeframe = kwargs['slow_timeframe']
    else:
        slow_timeframe = 'D'
    
    df = ds.f_df #just a short name
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    df['trendlines_diff_10_D'] = df['no_standing_upward_lines_10_D'] - df['no_standing_downward_lines_10_D']
    #features below look at the difference between current value of the above feature and its max/min
    df['trend_diff_change_up_D'] = (df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).min())
    df['trend_diff_change_down_D'] = -(df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).max())
    
    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI_' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).min ()
    df['max_RSI' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))
    
    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    preds[(df['RSI_' + fast_timeframe] < 30) & \
          (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe ] > 50) & \
          (df['trendlines_diff_10_D'] > 5) & \
          (df['trend_diff_change_down_D'] <= 3)] = 1
    preds[(df['RSI_' + fast_timeframe] < 30) & \
          (df['RSI_' + fast_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe] > 50) & \
          (df['trend_diff_change_up_D'] >= 5) & \
          (df['trendlines_diff_10_D'] > -5)] = 1.0
    
    #sells if:
    #opposite of above
    preds[(df['RSI_' + fast_timeframe] > 70) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50) & (df['trendlines_diff_10_D'] < -5) & (df['trend_diff_change_up_D'] <= 3)] = -1    
    preds[(df['RSI_' + fast_timeframe] > 70) & (df['RSI_' + fast_timeframe] > 30) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50) & (df['trend_diff_change_down_D'] >= 5) & (df['trendlines_diff_10_D'] < 5)] = -1.0

    return preds
    
def compute_predictions_v2 (ds):
    df = ds.f_df #just a short name
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    df['trendlines_diff_10_D'] = df['no_standing_upward_lines_10_D'] - df['no_standing_downward_lines_10_D']
    #features below look at the difference between current value of the above feature and its max/min
    df['trend_diff_change_up_D'] = (df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).min()) #how much off the lows
    df['trend_diff_change_down_D'] = -(df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).max()) # how much off the highs

    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI'] = df.RSI.rolling(window=10).min ()
    df['max_RSI'] = df.RSI.rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))

    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    preds[(df['RSI'] < 40) & (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe ] > 30) & (df['trendlines_diff_10_D'] > 5) & (df['trend_diff_change_down_D'] <= 5)] = 1
    preds[(df['RSI'] < 40) & (df['RSI'] < 70) & (df['RSI_' + slow_timeframe] < 70) & (df['RSI_' + slow_timeframe] > 30) & (df['trend_diff_change_up_D'] >= 5) & (df['trendlines_diff_10_D'] > -5)] = 1.0

    #sells if:
    #opposite of above
    preds[(df['RSI'] > 60) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 70) & (df['trendlines_diff_10_D'] < -5) & (df['trend_diff_change_up_D'] <= 5)] = -1    
    preds[(df['RSI'] > 60) & (df['RSI'] > 30) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 70) & (df['trend_diff_change_down_D'] >= 5) & (df['trendlines_diff_10_D'] < 5)] = -1.0

    return preds

def compute_predictions_pca (ds, **kwargs):
    df = ds.f_df #just a short name
    
    for key, value in kwargs.iteritems ():
        print (str(key) + ': ' + str (value))
        
    if 'fast_timeframe' in kwargs.keys ():
        fast_timeframe = kwargs['fast_timeframe']
    else:
        fast_timeframe = 'M15'
        
    if 'slow_timeframe' in kwargs.keys ():
        slow_timeframe = kwargs['slow_timeframe']
    else:
        slow_timeframe = 'D'
    
    sel_cols = []
    
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    df['trendlines_diff_10_' + slow_timeframe] = df['no_standing_upward_lines_10_' + slow_timeframe] - df['no_standing_downward_lines_10_' + slow_timeframe]
    sel_cols.append ('no_standing_downward_lines_10_' + slow_timeframe)
    sel_cols.append ('no_standing_upward_lines_10_' + slow_timeframe)
    sel_cols.append ('trendlines_diff_10_' + slow_timeframe)
    
    #features below look at the difference between current value of the above feature and its max/min
    df['trend_diff_change_up_' + slow_timeframe] = (df['trendlines_diff_10_' + slow_timeframe] - df['trendlines_diff_10_' + slow_timeframe].rolling(window=2000).min())
    sel_cols.append ('trend_diff_change_up_' + slow_timeframe)
    
    df['trend_diff_change_down_' + slow_timeframe] = - (df['trendlines_diff_10_' + slow_timeframe] - df['trendlines_diff_10_' + slow_timeframe ].rolling(window=2000).max())
    sel_cols.append ('trend_diff_change_down_' + slow_timeframe)
    
    #fixing a bug on the computation of normalized ratios
    try:
        df['close_over_100d_ma_normbyvol_' + slow_timeframe] = (df['Close_' + slow_timeframe] - \
                                                          df['ma_100_close_' + slow_timeframe]) / \
                                                          (df['hist_vol_3m_close_' + slow_timeframe] * \
                                                           df['ma_100_close_' + slow_timeframe])
    except:
        pass
                                                      
    sel_cols.append ('close_over_100d_ma_normbyvol_' + slow_timeframe)
    
    #adding RSI overbought and oversold indicators convolved with a 14 days window
    if 'avoid_overbought_slow' in kwargs.keys ():
        df['RSI_overbought_' + slow_timeframe] = df['RSI_' + slow_timeframe] > 70
        sel_cols.append ('RSI_overbought_' + slow_timeframe)
        sel_cols.append ('RSI_' + slow_timeframe)
        
        df['RSI_oversold_' + slow_timeframe] = df['RSI_' + slow_timeframe] < 30        
        sel_cols.append ('RSI_oversold_' + slow_timeframe)
        
        window_size = (14 if 'overbought_slow_window' not in kwargs.keys () else kwargs['overbought_slow_window']) * {'D': 24, 'H4': 4, 'H1': 1} [slow_timeframe] * {'D': 1.0/24.0, 'H4': 6.0/24.0, 'H1': 1.0, 'M15': 4.0} [fast_timeframe]
        df['RSI_overbought_' + slow_timeframe] = np.convolve(df['RSI_overbought_' + slow_timeframe], 
                      (np.ones (int(window_size)) if kwargs['avoid_overbought_slow'] else np.zeros ( int(window_size) )),
                      mode='full')[:len(df)]
        df['RSI_oversold_' + slow_timeframe] = np.convolve(df['RSI_oversold_' + slow_timeframe],
                  (np.ones ( int(window_size)) if kwargs['avoid_overbought_slow'] else np.zeros ( int(window_size) )),
                      mode='full')[:len(df)]
    else:
        df['RSI_overbought_' + slow_timeframe] = 0
        df['RSI_oversold_' + slow_timeframe] = 0
        
    if 'avoid_overbought_fast' in kwargs.keys ():
        df['RSI_overbought_' + fast_timeframe] = df['RSI_' + fast_timeframe] > 70
        df['RSI_oversold_' + fast_timeframe] = df['RSI_' + fast_timeframe] < 30        
        
        sel_cols.append ('RSI_' + fast_timeframe)
        sel_cols.append ('RSI_overbought_' + fast_timeframe)
        sel_cols.append ('RSI_oversold_' + fast_timeframe)
        
        window_size = (14 if 'overbought_fast_window' not in kwargs.keys () else kwargs['overbought_fast_window'])
        df['RSI_overbought_' + fast_timeframe] = np.convolve(df['RSI_overbought_' + fast_timeframe], 
                      (np.ones ( int(window_size) ) if kwargs['avoid_overbought_fast'] else np.zeros ( int(window_size) )), 
                      mode='full')[:len(df)]
        df['RSI_oversold_' + fast_timeframe] = np.convolve(df['RSI_oversold_' + fast_timeframe], 
                      (np.ones ( int(window_size) ) if kwargs['avoid_overbought_fast'] else np.zeros ( int(window_size) )),
                      mode='full')[:len(df)]
    else:
        df['RSI_overbought_' + fast_timeframe] = 0
        df['RSI_oversold_' + fast_timeframe] = 0
    #########################################################
    
    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI_' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).min ()
    df['max_RSI' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).max ()
    sel_cols.append ('min_RSI_' + fast_timeframe)
    sel_cols.append ('max_RSI_' + fast_timeframe)
    
    #computing predictions
    preds = np.zeros(len(ds.f_df))
    
    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    criteria = np.ones (len(ds.f_df))
    
    #minimum correlation to the first principal component
    if 'rho_min' in kwargs.keys (): 
        rho_min = kwargs['rho_min']
    else:
        rho_min = 0.3
        
    if 'rho_max' in kwargs.keys (): 
        rho_max = kwargs['rho_max']
    else:
        rho_max = 0.7
        
    #residual of the regression versus the first principal component
    if 'resid_min' in kwargs.keys (): 
        resid_min = kwargs['resid_min']
    else:
        resid_min = 5.0
        
    if 'resid_max' in kwargs.keys (): 
        resid_max = kwargs['resid_max']
    else:
        resid_max = 5.0
        
    #halflife
    if 'halflife_min' in kwargs.keys (): 
        halflife_min = kwargs['halflife_min']
    else:
        halflife_min = 150
    
    #maximum RSI fast value to allow to trigger long position
    if 'RSI_fast_max' in kwargs.keys (): 
        RSI_fast_max = kwargs['RSI_fast_max']
    else:
        RSI_fast_max = 50.0
        
    #minimum RSI fast value to allow to trigger long position
    if 'RSI_fast_min' in kwargs.keys (): 
        RSI_fast_min = kwargs['RSI_fast_min']
    else:
        RSI_fast_min = 30.0
        
    #maximum RSI slow value to allow to trigger long position
    if 'RSI_slow_max' in kwargs.keys (): 
        RSI_slow_max = kwargs['RSI_slow_max']
    else:
        RSI_slow_max = 70.0
        
    #minimum RSI slow value to allow to trigger long position
    if 'RSI_slow_min' in kwargs.keys (): 
        RSI_slow_min = kwargs['RSI_slow_min']
    else:
        RSI_slow_min = 50.0
        
    #net trendlines min: minimum difference between the number of upward lines and downward lines to trigger a long position
    if 'net_trendlines_min' in kwargs.keys (): 
        net_trendlines_min = kwargs['net_trendlines_min']
    else:
        net_trendlines_min = 5.0
        
    if 'net_trendlines_max' in kwargs.keys (): 
        net_trendlines_max = kwargs['net_trendlines_max']
    else:
        net_trendlines_max = 15.0
        
    #maximum number of upward trendlines broken or new downward lines to trigger long position
   
    if 'trendlines_delta_down_min' in kwargs.keys ():
        trendlines_delta_down_min = kwargs['trendlines_delta_down_min']
    else:
        trendlines_delta_down_min = 3.0
        
    if 'trendlines_delta_down_max' in kwargs.keys (): 
        trendlines_delta_down_max = kwargs['trendlines_delta_down_max']
    else:
        trendlines_delta_max = 8.0
        
    if 'trendlines_delta_up_min' in kwargs.keys ():
        trendlines_delta_up_min = kwargs['trendlines_delta_up_min']
    else:
        trendlines_delta_up_min = 3.0
        
    if 'trendlines_delta_up_max' in kwargs.keys (): 
        trendlines_delta_up_max = kwargs['trendlines_delta_up_max']
    else:
        trendlines_delta_up_max = 8.0
        
    if 'trendlines_delta_min' in kwargs.keys ():
        trendlines_delta_up_min = trendlines_delta_down_min = kwargs['trendlines_delta_min']    
    else:
        trendlines_delta_up_min = trendlines_delta_down_min = 3.0
    
    if 'trendlines_delta_max' in kwargs.keys ():
        trendlines_delta_up_max = trendlines_delta_down_max = kwargs['trendlines_delta_max']
    else:
        trendlines_delta_up_max = trendlines_delta_down_max = 8.0
        
    #close_over_100d_ma_normbyvol_D    
    if 'close_over_ma_min' in kwargs.keys ():
        close_over_ma_min = kwargs['close_over_ma_min']
    else:
        close_over_ma_min = 1.0
        
    if 'close_over_ma_max' in kwargs.keys ():
        close_over_ma_max = kwargs['close_over_ma_max']
    else:
        close_over_ma_max = 5.0
    
    if 'criterium' in kwargs.keys ():
        criterium = kwargs['criterium']
    else:
        criterium = 'both'
        
    #completes the dataframe in order to produce some results quickly
    if 'rho_' + PCA_SUFFIX + fast_timeframe not in df.columns:
        print ('PCA features not found - adding some dummy data to the dataframe')
        df['rho_' + PCA_SUFFIX + fast_timeframe] = 0.5 * np.ones (len(df))
        df['n_resid_' + PCA_SUFFIX + fast_timeframe] = np.zeros (len(df))
        
    sel_cols.append ('rho_' + PCA_SUFFIX + fast_timeframe)
    sel_cols.append ('n_resid_' + PCA_SUFFIX + fast_timeframe)
    sel_cols.append ('halflife_' + slow_timeframe)
    sel_cols.append ('Close_' + fast_timeframe)
    sel_cols.append ('Open_' + fast_timeframe)
    sel_cols.append ('High_' + fast_timeframe)
    sel_cols.append ('Low_' + fast_timeframe)
    sel_cols.append ('hist_vol_1m_close_' + fast_timeframe)
    
    for col in df.columns:
        if col not in sel_cols:
            del df[col]
        
    #buys when balance of trendlines moves up, momentum    
    if criterium == 'first' or criterium == 'both':
        preds[ ((np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) > rho_min) & \
              (np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) < rho_max) & \
              ((df['close_over_100d_ma_normbyvol_' + slow_timeframe]) > close_over_ma_min) & \
              ((df['close_over_100d_ma_normbyvol_' + slow_timeframe]) < close_over_ma_max) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] < - resid_min) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] > - resid_max) if 'rho_' + PCA_SUFFIX + fast_timeframe in df.columns else np.full (len(preds), True))& \
              (df['RSI_' + fast_timeframe] < RSI_fast_max) & \
              (df['RSI_' + fast_timeframe] > RSI_fast_min) & \
              (df['RSI_' + slow_timeframe] < RSI_slow_max) & \
              (df['RSI_' + slow_timeframe ] > RSI_slow_min) & \
              ((df['halflife_' + slow_timeframe] > halflife_min) | (df['halflife_' + slow_timeframe ] < 0)) &\
              (df['trendlines_diff_10_' + slow_timeframe] >= net_trendlines_min) & \
              (df['trendlines_diff_10_' + slow_timeframe] <= net_trendlines_max) & \
              #(df['trend_diff_change_down_' + slow_timeframe] >= trendlines_delta_down_min) &\
              #(df['trend_diff_change_down_' + slow_timeframe] <= trendlines_delta_down_max) &\
              (df['RSI_overbought_' + slow_timeframe] == 0) &\
              (df['RSI_overbought_' + fast_timeframe] == 0) &\
              #(df['RSI_oversold_' + slow_timeframe] > 0) &\
              (df['trend_diff_change_up_' + slow_timeframe] >= trendlines_delta_up_min) &\
              (df['trend_diff_change_up_' + slow_timeframe] <= trendlines_delta_up_max)
              ] = 1
        
    #sells if:
    #opposite of above    
        preds[((np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) > rho_min) & \
              (np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) < rho_max) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] > resid_min) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] < resid_max) if 'rho_' + PCA_SUFFIX + fast_timeframe in df.columns else np.full (len(preds), True)) & \
              ((df['close_over_100d_ma_normbyvol_' + slow_timeframe]) < -close_over_ma_min) & \
              ((df['close_over_100d_ma_normbyvol_' + slow_timeframe]) > -close_over_ma_max) & \
              (df['RSI_' + fast_timeframe] > (100.0 - RSI_fast_max)) & \
              (df['RSI_' + fast_timeframe] < (100.0 - RSI_fast_min)) & \
              (df['RSI_' + slow_timeframe] > (100.0 - RSI_slow_max)) & \
              (df['RSI_' + slow_timeframe] < (100.0 - RSI_slow_min)) & \
              ((df['halflife_' + slow_timeframe ] > halflife_min) | (df['halflife_' + slow_timeframe ] < 0)) &\
              (df['trendlines_diff_10_' + slow_timeframe] <= -net_trendlines_min) & \
              (df['trendlines_diff_10_' + slow_timeframe] >= -net_trendlines_max) & \
              #(df['RSI_overbought_' + slow_timeframe] > 0) &\
              (df['RSI_oversold_' + slow_timeframe] == 0) &\
              (df['RSI_oversold_' + fast_timeframe] == 0) &\
              (df['trend_diff_change_down_' + slow_timeframe] >= trendlines_delta_up_min) &\
              (df['trend_diff_change_down_' + slow_timeframe] <= trendlines_delta_up_max)
              #(df['trend_diff_change_up_' + slow_timeframe] >= trendlines_delta_down_min) &\
              #(df['trend_diff_change_up_' + slow_timeframe] <= trendlines_delta_down_max)
              ] = -1 
    
    #buys when downward lines are broken upwards
    if criterium == 'second' or criterium == 'both':      
        preds[ (np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) > rho_min) & \
              (np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) < rho_max) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] < - resid_min) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] > - resid_max) & \
              (df['RSI_' + fast_timeframe] < RSI_fast_max) & \
              (df['RSI_' + fast_timeframe] > RSI_fast_min) & \
              (df['RSI_' + slow_timeframe] < RSI_slow_max) & \
              (df['RSI_' + slow_timeframe ] > RSI_slow_min) & \
              (df['trendlines_diff_10_' + slow_timeframe] >= net_trendlines_min) & \
              (df['trendlines_diff_10_' + slow_timeframe] <= net_trendlines_max) & \
              (df['trend_diff_change_down_' + slow_timeframe] >= trendlines_delta_down_min) &\
              (df['trend_diff_change_down_' + slow_timeframe] <= trendlines_delta_down_max)
              #(df['trend_diff_change_up_' + slow_timeframe] >= trendlines_delta_up_min) &\
              #(df['trend_diff_change_up_' + slow_timeframe] <= trendlines_delta_up_max)
              ] = 1
        
      #sells if:
      #opposite of above
        preds[(np.abs(df['rho_' +PCA_SUFFIX + fast_timeframe]) > rho_min) & \
              (np.abs(df['rho_' + PCA_SUFFIX + fast_timeframe]) < rho_max) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] > resid_min) & \
              (df['n_resid_' + PCA_SUFFIX + fast_timeframe] < resid_max) & \
              (df['RSI_' + fast_timeframe] > (100.0 - RSI_fast_max) ) & \
              (df['RSI_' + fast_timeframe] < (100.0 - RSI_fast_min)) & \
              (df['RSI_' + slow_timeframe] > (100.0 - RSI_slow_max)) & \
              (df['RSI_' + slow_timeframe] < (100.0 - RSI_slow_min)) & \
              (df['trend_diff_change_down_' + slow_timeframe] >= trendlines_delta_max) & \
              (df['trendlines_diff_10_' + slow_timeframe] <= net_trendlines_max) & \
              (df['trendlines_diff_10_' + slow_timeframe] >= net_trendlines_min)] = -1.0

    return preds

class VectorizedStrategy ():
    def __init__ (self, timeframe='M15', 
                  other_timeframes=['D'],
                  from_time=2006, 
                  to_time=2015,
                  ds=None):
        
        self.from_time = None
        self.to_time = None
        set_from_to_times (self, from_time, to_time)
        
        self.timeframe = None #trading timeframe
        self.other_timeframes = None
        
        self.pnl_df = None
        self.init_timeframes (timeframe, other_timeframes)
        self.dsh = None
        self.hit_miss_cache = {}
        
    def init_timeframes (self, timeframe, other_timeframes):
        if timeframe is not None:
            self.timeframe = timeframe
        if other_timeframes is not None:
            self.other_timeframes = other_timeframes
        self.init_pnl_dataframe ()
        
    def init_instrument (self, instrument = None):
        if instrument is not None:
            self.instrument = instrument
            if self.dsh is not None:
                self.dsh.init_instrument (instrument)
                if instrument + '_' + self.timeframe in self.dsh.ds_dict.keys ():
                    self.ds = self.dsh.ds_dict [instrument + '_' + self.timeframe]
        
    def init_pnl_dataframe (self):
        self.pnl_df = None
        #init pnl dataframe
        ds = Dataset(ccy_pair='GBP_USD', 
                     from_time=self.from_time, 
                     to_time=self.to_time, timeframe=self.timeframe)
        ds.loadCandles()
        self.pnl_df = pd.core.frame.DataFrame(index=ds.df.index)
        
    def load_instrument (self, instrument='USD_ZAR',
                         timeframe=None, 
                         other_timeframes=None,
                         slow_timeframe_delay = 5):
        self.init_timeframes (timeframe, other_timeframes)
        self.init_instrument (instrument)
        
        if self.dsh is None:
            self.dsh = DatasetHolder(from_time =self.from_time, 
                                 to_time=self.to_time, 
                                 instrument=instrument)
        else:
            self.dsh.init_instrument (self.instrument)
        
        timeframe_list = self.other_timeframes + [self.timeframe]
        
        for timeframe in timeframe_list:
            if instrument + '_' + timeframe not in self.dsh.ds_dict.keys ():        
                self.dsh.loadMultiFrame(ccy_pair_list = [instrument], 
                                        timeframe_list = timeframe_list, 
                                        bComputeFeatures=[tf != 'D' for tf in timeframe_list], 
                                        bLoadFeatures=[tf == 'D' for tf in timeframe_list])
                try:
                    #loads PCA features
                    self.dsh.ds_dict[instrument+'_'+self.timeframe].loadPCAFeatures ()                        
                except:
                    pass
                
                if True:
                    #computes additional features
                    #halflife
                    aux_ds = self.dsh.ds_dict[self.instrument+'_' + self.other_timeframes[0]]
                    b = np.array([halflife (aux_ds.f_df.Close[i-252:i]) for i in range(252, len(aux_ds.f_df))])
                    aux_ds.f_df['halflife'] = np.zeros(len(aux_ds.f_df))
                    aux_ds.f_df['halflife'][252:] = b[:,1]
                #except:
                #    print ('Failed to compute feature: halflife')
                
                self.ds = self.dsh.ds_dict[instrument+'_'+self.timeframe]
        
                for htf in self.other_timeframes:
                    self.dsh.appendTimeframesIntoOneDataset (lower_timeframe=self.timeframe, 
                                                             higher_timeframe=htf,
                                                         daily_delay=slow_timeframe_delay)
                break
        
    def compute_predictions (self, **kwargs):
        if 'func' in kwargs.keys ():
            func = kwargs['func']
            if type (func) == str:
                try:
                    func = eval (func)
                except:
                    print ('Failed to evaluate function, defaulting to compute_predictions_pca')
                    func = compute_predictions_pca
        else:
            func = compute_predictions_pca
        preds = func(self.ds, **kwargs)
        self.ds.set_predictions(preds)
        
        return self
    
    def summary (self):
        pass
        
    def plot_pnl (self, bMultiple = False, bSave=False, label = '', plot_filename=''):
        return plot_pnl(self.ds, bMultiple = bMultiple, bSave=bSave,
                        label = label, plot_filename = plot_filename)
                
    def plot_signals (self, plot_filename=''):
        return plot_signals(self.ds, plot_filename=plot_filename)
                
    def plot_hist (self, plot_filename=''):
        return plot_histogram(self.ds, plot_filename=plot_filename)
    
    def load_multiple (self):
        pass
    
    def compute_pred_multiple (self, **kwargs):
        try:
            self.preds_hash_table_dict.keys ()
        except:
            self.preds_hash_table_dict = {}
            
        instruments = self.dsh.getLoadedInstruments ()
        
        old_ds = self.ds
        for instrument in instruments:
            if instrument in self.preds_hash_table_dict.keys ():
                if self.preds_hash_table_dict [instrument] == sha1 (str(kwargs)).hexdigest ():
                    print (instrument + ' predictions cached')
                    continue
            print (instrument + ' predictions being calculated')
            self.ds = self.dsh.ds_dict[instrument + '_' + self.timeframe]
            self.compute_predictions (**kwargs)
            
            if 'serial_gap' in kwargs.keys ():
                serial_gap = (kwargs['serial_gap'])
            else:
                serial_gap = 0
            if serial_gap > 0:
                self.ds.removeSerialPredictions (serial_gap)
            self.preds_hash_table_dict [instrument] = sha1 (str(kwargs)).hexdigest ()
        self.ds = old_ds
            
    def compute_hit_miss (self):
        instruments = self.dsh.getLoadedInstruments ()
        
        for instrument in instruments:
            if instrument not in self.hit_miss_cache.keys ():            
                self.hit_miss_cache [instrument] = {}
                hit_miss_array = self.hit_miss_cache [instrument] ['hit_miss_array'] = compute_hit_miss_array(self.dsh.ds_dict[instrument + '_' + self.timeframe])
                self.hit_miss_cache [instrument] ['pnl'] = np.maximum(hit_miss_array, 0) * self.dsh.ds_dict[instrument+'_'+self.timeframe].target_multiple + np.minimum (hit_miss_array, 0) * 1
            hit_miss_array = self.hit_miss_cache [instrument] ['hit_miss_array']
            pnl = np.cumsum(self.hit_miss_cache [instrument] ['pnl'])
            self.pnl_df[instrument] = pnl
        if 'Total' in self.pnl_df.columns:
            self.pnl_df['Total'] = np.zeros (len(self.pnl_df))
        self.pnl_df['Total'] = self.pnl_df.dropna().sum(axis=1)
    
    def plot_multiple_pnl (self, bPlotSum = True):
        self.compute_hit_miss ()
        
        instruments = self.dsh.getLoadedInstruments ()
        
        
        fig = plt.figure ()        
        plt.title('PnL')
        
        for instrument in instruments:
            pnl = self.hit_miss_cache [instrument] ['pnl']
            
            if not bPlotSum:
                plt.plot (self.pnl_df[instrument].dropna(), label=instrument)
                plt.legend (loc='best')
                
        if bPlotSum:
            plt.plot (self.pnl_df['Total'].dropna(), label='Total')
            plt.legend (loc='best')
                
        return fig        
    
    def summarize_stats (self):
        instruments = self.dsh.getLoadedInstruments ()
        self.compute_hit_miss ()
        
        self.strat_summary = {}
        
        self.strat_summary['total_trades'] = 0
        self.strat_summary['total_pnl'] = 0
        outstr = '#########################################################\n'
        
        total_trades = 0
        total_pnl = 0
        outstr += 'Instrument\tMin Stop\tTgt x\tTrades\tHit Ratio\tPnL\n'
        for instrument in instruments:
            ds = self.dsh.ds_dict[instrument+'_'+self.timeframe]
            
            hit_miss_array = self.hit_miss_cache [instrument] ['hit_miss_array']
            pnl = np.cumsum(self.hit_miss_cache [instrument] ['pnl'])[-1]
            trades = np.count_nonzero(hit_miss_array!=0)
            try:
                hit_ratio = float(np.sum(hit_miss_array)) / float(trades) / 2 + 0.5
            except:
                hit_ratio = np.nan
            
            self.strat_summary [instrument] = {
                    'from_time': ds.f_df.index[0],
                    'to_time': ds.f_df.index[-1],
                    'min_stop': ds.min_stop,
                   'target_multiple': ds.target_multiple,
                   'no_trades' : trades,
                   'hit_ratio' : hit_ratio,
                   'pnl' : pnl}
            outstr += '%s\t%.3f\t%.1f\t%d\t%.2f\t%d\n'%(instrument, ds.min_stop, 
                                                        ds.target_multiple, 
                                                        trades, hit_ratio, pnl)
            
            total_trades += trades
            total_pnl += pnl
            self.strat_summary['total_trades'] += trades
            self.strat_summary['total_pnl'] += pnl
            
        outstr += '%s\t%d\t%.2f\t%d\n'%('Total', total_trades, float(total_pnl) / float(total_trades) / 2 + 0.5, total_pnl)
        
        #calc concentration, max drawdown per pair
        concentration = np.sum([self.pnl_df[col][-1]**2 for col in self.pnl_df.columns])/total_pnl ** 2
        outstr += '\nConcentration index: %.2f\n\n'%(concentration)
        
        #computes max drawdown
        xs = self.pnl_df['Total'].dropna ()        
        i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
        j = np.argmax(xs[:i]) # start of period
        max_drawdown = xs[j] - xs[i]
        k = np.argmin( (xs[i:] - xs[j])**2 )
        
        outstr += 'Max drawdown\tMax pct\tMax pct Total\tLength\tRecovery\n'
        outstr += '\t%d\t%.2f\t%.2f\t%d\t%.0f\n'%(max_drawdown, 
                                    float (max_drawdown/xs[j]), 
                                    float (max_drawdown/total_pnl),
                                    (i-j).days,
                                    (float((k-i).days) if xs[k] >= xs[j] else np.nan)
                                    )
        print (outstr)
        
        return outstr           
        
if False:
    plots_path = u'./Analysis/Results/Strategies/Vectorized/Trendlines_and_change_RSI_2_timeframes'
    ccy = 'AUD_USD'
    slow_timeframe = 'D'
    fast_timeframe_list = ['M15']
    daily_delay = 5     #to avoid look-ahead bias
    serial_gap_list = [0] #, 20, 80, 160]   #to remove serial predictions
    from_time = 2004
    to_time = 2014
    
    ds = Dataset(ccy_pair='EUR_USD', from_time=2006, to_time=2015)
    ds.loadCandles()
    
    fast_timeframe = 'M15'
    pnl = pd.core.frame.DataFrame(index=ds.df.index)
    for ccy in full_instrument_list:
        try:
            dsh = DatasetHolder(from_time =2006, to_time=2015, instrument=ccy)
            dsh.loadMultiFrame(ccy_pair_list=[ccy], timeframe_list=['D', 'M15'], bComputeFeatures=[False, True], bLoadFeatures=[True, False])
            dsh.ds_dict[ccy+'_M15'].loadPCAFeatures ()
            ds_d = dsh.ds_dict[ccy+'_D']
            ds = dsh.ds_dict[ccy+'_M15']
    
            dsh.appendTimeframesIntoOneDataset (lower_timeframe='M15', daily_delay=0)
            
            preds = compute_predictions(ds)
            ds.set_predictions(preds)
            #ds.removeSerialPredictions()
            plot_pnl(ds)
            pnl[ccy] = np.cumsum(ds.l_df.Labels * ds.p_df.Predictions)[pnl.index]
        except:
            pass
    plt.plot(pnl.sum(axis=1))

if False:
    for fast_timeframe in fast_timeframe_list:

        for ccy in full_instrument_list:
            try:
                dsh = DatasetHolder(from_time=from_time, 
                                    to_time=to_time, 
                                    instrument=ccy)
                dsh.loadMultiFrame ()
                dsh.appendTimeframesIntoOneDataset(lower_timeframe=fast_timeframe,
                                                   higher_timeframe=slow_timeframe)
                
                ds_f = dsh.ds_dict[ccy+'_'+fast_timeframe]
                preds = compute_predictions (ds_f)
                
                for serial_gap in serial_gap_list:
                    ds_f.set_predictions(preds) #uses deepcopy, creates ds.p_df
                    if serial_gap != 0:
                        ds_f.removeSerialPredictions(serial_gap)
                
                    plot_signals (ds_f, True, plots_path + '/Signals_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_pnl (ds_f, True, plots_path + '/PnL_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_histogram (ds_f, True, plots_path + '/Histogram_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                
                
            except:
                pass
            
if False:
    for fast_timeframe in fast_timeframe_list:
        for ccy in fx_list:

            try:
                ds = Dataset (ccy_pair = ccy, 
                              timeframe = fast_timeframe,
                              from_time = from_time,
                              to_time = to_time)
                ds.loadFeatures ()
                preds = compute_predictions (ds)
            except:
                try:                
                    dsh = DatasetHolder(from_time=from_time, 
                                    to_time=to_time, instrument=ccy)
                    try:
                        dsh.loadMultiFrame(timeframe_list=[slow_timeframe, fast_timeframe])
                        ds = dsh.ds_dict[ccy + '_' + fast_timeframe]
                        ds.computeFeatures (bComputeIndicators=True,
                                                 bComputeNormalizedRatios=True,
                                                 bComputeCandles=False,
                                                 bComputeHighLowFeatures=False)
                    except:                        
                        ds = Dataset (ccy_pair = ccy, 
                              timeframe = fast_timeframe,
                              from_time = from_time,
                              to_time = to_time)
                        ds.computeFeatures (bComputeIndicators=True,
                                                 bComputeNormalizedRatios=True,
                                                 bComputeCandles=False,
                                                 bComputeHighLowFeatures=False)
                        ds.saveFeatures ()
                        try:
                            ds.loadLabels ()
                            assert (len(ds.f_df) == len (ds.l_df))
                        except:
                            ds.computeLabels ()
                            sa.saveLabels ()
                        dsh.loadMultiFrame(timeframe_list=[slow_timeframe, fast_timeframe])
                        
                    
                    
                    
                    #appends the slow timeframe columns to the fast timeframe one
                    dsh.appendTimeframesIntoOneDataset(instrument = ccy, 
                                                       lower_timeframe = fast_timeframe,
                                                       daily_delay=daily_delay)
                    ds = dsh.ds_dict[ccy + '_' + fast_timeframe]
                    ds.saveFeatures ()
                    preds = compute_predictions (ds)
                except:
                    pass
                    
            try:
                ds.loadLabels ()
                labels = ds.get_active_labels ()
                
                assert (len(ds.f_df) == len (ds.l_df))

            except:
                ds.computeLabels ()
                ds.saveLabels ()
                
            try:    
                for serial_gap in serial_gap_list:
                    ds.set_predictions(preds) #uses deepcopy, creates ds.p_df
                    if serial_gap != 0:
                        ds.removeSerialPredictions(serial_gap)
                
                    plot_signals (ds, True, plots_path + '/Signals_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_pnl (ds, True, plots_path + '/PnL_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_histogram (ds, True, plots_path + '/Histogram_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
            except:
                print ('An error ocurred: ' + ccy + fast_timeframe)