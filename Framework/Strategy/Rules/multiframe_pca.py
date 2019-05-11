# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np

def mtf_pca (ds, **kwargs):
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