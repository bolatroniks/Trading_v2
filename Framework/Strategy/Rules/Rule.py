# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
#import xgboost as xgb
#from xgboost import XGBClassifier
from sklearn import metrics   #Additional scklearn functions

try:
    from sklearn import cross_validation
except:
    from sklearn.model_selection import cross_validate as cross_validation

try:
    from sklearn.grid_search import GridSearchCV, RandomizedSearchCV   #Perforing grid search
except:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#from xgboost import plot_tree
from matplotlib.pylab import rcParams
from sklearn.preprocessing import label_binarize

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from dateutil.relativedelta import relativedelta

#from Framework.Training.Dataset import Dataset
#from Framework.Training.TradingModel import TradingModel
from Config.const_and_paths import NEUTRAL_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL

try:
    from Framework.Features.Features import *
except:
    pass
import operator


from Config.const_and_paths import *

def default_stop (my_df, bLive=True, 
                  vol_denominator = 5.0, min_target = 0.01, 
                  idx=-1,  #last prediction, default for live mode                  
                  target_multiple=1.0):
    vol_hist_col = [_ for _ in my_df.columns if _.find ('hist_vol_1m_close') >= 0] [0]
    if bLive:
        stop = target_multiple * np.maximum(my_df[vol_hist_col][idx] / vol_denominator, min_target)
    else:
        #to be implemented
        return 0.01
    return stop
    
default_target = default_stop

def default_filter_instrument (ccy_pair, last_timestamp, **kwargs):
    return True    

class Rule ():
    def __init__ (self, name='', func='', filter_instrument_func = default_filter_instrument,
                  args = {}, ruleType='SingleDataset',
                  stop_fn = None, target_fn = None, target_multiple=1.0, 
                  bUseHighLowFeatures= False,
                  timeframe = 'M15',
                  other_timeframes = ['D']):
        self.name = name
        self.func = func
        if filter_instrument_func is None:
            self.filter_instrument_func = default_filter_instrument
        else:
            self.filter_instrument_func = filter_instrument_func
        self.args = args
        self.ruleType = ruleType
        self.stop_fn = None
        self.target_fn = None
        self.pred = None
        self.target_multiple = target_multiple #in case target and stop use the same fn
        
        self.init_stop_target_fn (stop_fn, target_fn)
        
        self.timeframe = timeframe
        self.other_timeframes = other_timeframes
        self.bUseHighLowFeatures = bUseHighLowFeatures
        
    def init_stop_target_fn (self, stop_fn=None, target_fn=None):
        if stop_fn is not None:
            self.stop_fn = stop_fn
        if self.stop_fn is None:
            self.stop_fn = default_stop
            
        if target_fn is not None:
            self.target_fn = target_fn
        if self.target_fn is None:
            self.target_fn = default_target
            
    def filter_instruments (self, ccy_pair, last_timestamp):
        if self.filter_instrument_func is None:
            return True
        return self.filter_instrument_func (ccy_pair, last_timestamp, **self.args)
        
    def predict (self, obj, verbose=False):
        try:
            self.pred = self.func (obj, self.args)
        except TypeError:
            self.pred = self.func (obj, **self.args)
        return self.pred
        
    def get_stop_target (self, obj, verbose=False):
        stop = self.stop_fn (obj)
        target = self.target_fn (obj) * self.target_multiple
        
        return stop, target

def evaluate_rule (predictions, labels, verbose=False, 
                   plotPred=False,
                   feat_sel='Close'):
    pred = label_binarize(predictions - NEUTRAL_SIGNAL + 1, classes=[0,1,2])
    acc = metrics.accuracy_score(labels[predictions!=NEUTRAL_SIGNAL], pred[predictions!=NEUTRAL_SIGNAL])
    neutral_ratio = np.float(len(pred[predictions==NEUTRAL_SIGNAL])) / np.float(len(pred))
    if verbose:
        print ('Accuracy: '+str(acc) +', neutral%: '+str(neutral_ratio))
        
    if plotPred:
        #preds = t_xgb.model.predict(xgb.DMatrix(cv_X))        
        fig, ax1 = plt.subplots()
        
        ax2 = ax1.twinx()
        ax1.plot(predictions, 'g-', label='Predictions')
        #ax2.plot(ds.X[:,-1,ds.getFeatIdx(feat_sel)], label=feat_sel)
        
        ax1.set_xlabel('X data')
        ax1.set_ylabel('Y1 data', color='g')
        ax2.set_ylabel('Y2 data')
        plt.legend ()
        plt.show()
    return acc, neutral_ratio


def relabel_custom (X, y, idx=[179,180]):
    for i in range(len(X)):
        if X[i,-1, idx[0]] >= 3 and y[i,0] == 1:
            y[i,0] = 1
        else:
            y[i,0] = 0
            y[i,1] = 1
            y[i,2] = 0

        if X[i,-1, idx[1]] >= 3 and y[i,2] == 1:
            y[i,2] = 1
        else:
            y[i,2] = 0
            y[i,1] = 1
            y[i,0] = 0

    return y
    
def rule_SimpleMomentum(ds, 
                        threshold=0.01, 
                        criterium='close_over_100d_ma', 
                        verbose=False):
    if verbose:
        print('simple momentum rule, no filters')
    idx_ratio = ds.getFeatIdx(criterium)
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        if ds.X[i,-1,idx_ratio] > 1 + threshold:
            predictions [i] = 2
        elif ds.X[i,-1,idx_ratio] < 1 - threshold:
            predictions [i] = 0
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose)
    
def rule_SimpleMomentumWithVIXFilter (ds, threshold=0.01, criterium='close_over_100d_ma', 
                                      vix_criterium='VIX_close_over_21d_ma', 
                                      vix_ratio_threshold=1.1, verbose=False):
    if verbose:
        print ('simple momentum rule with VIX filter, predicts neutral if VIX closes above moving average')
    
    idx_ratio = ds.getFeatIdx(criterium)
    idx_vix_over_ma = ds.getFeatIdx(vix_criterium)
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and (ds.X[i,-1,idx_vix_over_ma] < vix_ratio_threshold):
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and (ds.X[i,-1,idx_vix_over_ma] < vix_ratio_threshold):
            predictions [i] = 0
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose)
    
def rule_SimpleMomentumWithVolFilter (ds, threshold=0.01, criterium='close_over_100d_ma', 
                                      vol_criterium='VIX_close_over_21d_ma', 
                                      vol_ratio_threshold=1.1,
                                      verbose=False):
    if verbose:
        print ('simple momentum rule with vol filter, buys/sells only when vol picks up')    
    idx_ratio = ds.getFeatIdx(criterium)
    idx_vol_ratio = ds.getFeatIdx(vol_criterium)
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold):
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold):
            predictions [i] = 0
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose)
    
def rule_SimpleMomentumWithTrendingFilter (ds, threshold=0.01, criterium='close_over_100d_ma', 
                                      trending_criterium='ratio_standing_up_downward_lines_10', 
                                      trending_ratio_threshold=2.0,
                                      verbose=False):    
    if verbose:
        print('simple momentum rule with trending filter, predicts neutral if ratio of upward over downward lines is close to 1')
    
    idx_ratio = ds.getFeatIdx(criterium)
    idx_lines_ratio = ds.getFeatIdx(trending_criterium)    
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and (ds.X[i,-1,idx_lines_ratio] > trending_ratio_threshold):
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and (ds.X[i,-1,idx_lines_ratio] < (1.0/trending_ratio_threshold)):
            predictions [i] = 0
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose)
    
def rule_ComplexMomentumWithTrendingFilter (ds, threshold=0.01, criterium='close_over_100d_ma', 
                                      trending_criterium='ratio_standing_up_downward_lines_10', 
                                      trending_ratio_threshold=2.0,
                                      vol_ratio_threshold=1.1,
                                      dist_criterium_low='dist_standing_low_30',
                                      dist_criterium_high='dist_standing_high_30',
                                      dist_threshold=0.015,
                                      vol_criterium='2w_over_3m_vol_ratio',
                                      verbose=False): 
    #simple momentum rule with trending filter, predicts neutral if ratio of upward over downward lines is close to 1
    if verbose:
        print ('Complex momentum rule with trending filter + minimum distance to high/low requirement')
    
    predictions = np.zeros (ds.X.shape[0])
    idx_ratio = ds.getFeatIdx(criterium)
    idx_lines_ratio = ds.getFeatIdx(trending_criterium)
    idx_low_dist = ds.getFeatIdx(dist_criterium_low)
    idx_high_dist = ds.getFeatIdx(dist_criterium_high)    
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and (ds.X[i,-1,idx_lines_ratio] > trending_ratio_threshold) and (np.abs(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0])<=dist_threshold):
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and (ds.X[i,-1,idx_lines_ratio] < (1.0 / trending_ratio_threshold)) and (np.abs(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0])<=dist_threshold):
            predictions [i] = 0
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose)
    
def rule_complexRule (ds, threshold=0.01, criterium='close_over_100d_ma', 
                                      trending_criterium='ratio_standing_up_downward_lines_10', 
                                      trending_ratio_threshold=2.0,
                                      dist_criterium_low='dist_relevant_low_30',
                                      dist_criterium_high='dist_relevant_high_30',
                                      dist_threshold=0.015,                                      
                                      vol_criterium='2w_over_3m_vol_ratio',
                                      vol_ratio_threshold=1.1,
                                      osc_criterium='RSI',
                                      osc_threshold_high=70,
                                      osc_threshold_low=30,
                                      verbose=False,
                                      plotPred=False):
    if verbose:        
        print ('complex rule')
    
    predictions = np.zeros (ds.X.shape[0])
    idx_ratio = ds.getFeatIdx(criterium)
    idx_vol_ratio = ds.getFeatIdx(vol_criterium)
    idx_lines_ratio = ds.getFeatIdx(trending_criterium)
    idx_low_dist = ds.getFeatIdx(dist_criterium_low)
    idx_high_dist = ds.getFeatIdx(dist_criterium_high)    
    
    for i in range (len(predictions)):
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and \
            (ds.X[i,-1,idx_lines_ratio] > trending_ratio_threshold) and \
            (np.abs(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0])<=dist_threshold) and \
            (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold):
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and \
                (ds.X[i,-1,idx_lines_ratio] < (1.0/trending_ratio_threshold)) and \
                (np.abs(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0])<=dist_threshold) and \
                (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold):
            predictions [i] = 0
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose, plotPred)

def rule_complexRuleWithRSI (ds, threshold=0.01, criterium='close_over_100d_ma', 
                      trending_criterium='ratio_standing_up_downward_lines_10', 
                      trending_ratio_threshold=2.0,
                      dist_criterium_low='dist_relevant_low_30',
                      dist_criterium_high='dist_relevant_high_30',
                      dist_threshold=0.015,                                      
                      vol_criterium='2w_over_3m_vol_ratio',
                      vol_ratio_threshold=1.1,
                      osc_criterium='RSI',
                      osc_threshold_high=70,
                      osc_threshold_low=30,
                      verbose=False,
                      plotPred=False):
    if verbose:        
        print ('complex rule with RSI')

    idx_ratio = ds.getFeatIdx(criterium)
    
    idx_50dma = ds.getFeatIdx('close_over_50d_ma')
    idx_100dma = ds.getFeatIdx('close_over_100d_ma')
    idx_200dma = ds.getFeatIdx('close_over_200d_ma')
    
    idx_vol_ratio = ds.getFeatIdx(vol_criterium)
    idx_lines_ratio = ds.getFeatIdx(trending_criterium)
    idx_low_dist = ds.getFeatIdx(dist_criterium_low)
    idx_high_dist = ds.getFeatIdx(dist_criterium_high)
    idx_rsi = ds.getFeatIdx(osc_criterium)    
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        dist_low = np.abs(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0])
        dist_high = np.abs(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0])
        
        dist_50dma = ds.X[i,-1,idx_50dma] - 1.0
        dist_100dma = ds.X[i,-1,idx_100dma] - 1.0
        dist_200dma = ds.X[i,-1,idx_200dma] - 1.0

        min_dist_l = np.min([np.abs(dist_100dma), np.abs(dist_200dma), dist_low])
        #min_dist_l = np.abs(dist_low)
        #min_dist_h = np.abs(dist_high)
        min_dist_h = np.min([np.abs(dist_100dma), np.abs(dist_200dma), dist_high])
        
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and \
            (ds.X[i,-1,idx_lines_ratio] > trending_ratio_threshold) and \
            (min_dist_l<=dist_threshold) and \
            (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold) and \
            (ds.X[i,-1,idx_rsi] <= osc_threshold_high):
            #print (str(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0]))
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and \
                (ds.X[i,-1,idx_lines_ratio] < (1.0/trending_ratio_threshold)) and \
                (min_dist_h<=dist_threshold) and \
                (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold) and \
                (ds.X[i,-1,idx_rsi] >= osc_threshold_low):
            predictions [i] = 0
            #print (str(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0]))
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose, plotPred)
    
def rule_veryComplex (ds, threshold=0.01, criterium='close_over_100d_ma', 
                                      trending_criterium='ratio_standing_up_downward_lines_10', 
                                      trending_ratio_threshold=2.0,
                                      dist_criterium_low='dist_relevant_low_30',
                                      dist_criterium_high='dist_relevant_high_30',
                                      dist_threshold=0.015,                                      
                                      vol_criterium='2w_over_3m_vol_ratio',
                                      vol_ratio_threshold=1.1,
                                      osc_criterium='RSI',
                                      osc_threshold_high=70,
                                      osc_threshold_low=30,
                                      verbose=False,
                                      plotPred=False):
    if verbose:
        print ('complex rule with RSI and dist 21dma')

    idx_ratio = ds.getFeatIdx(criterium)
    
    idx_21dma = ds.getFeatIdx('close_over_21d_ma')
    idx_50dma = ds.getFeatIdx('close_over_50d_ma')
    idx_100dma = ds.getFeatIdx('close_over_100d_ma')
    idx_200dma = ds.getFeatIdx('close_over_200d_ma')
    
    idx_vol_ratio = ds.getFeatIdx(vol_criterium)
    idx_lines_ratio = ds.getFeatIdx(trending_criterium)
    idx_low_dist = ds.getFeatIdx(dist_criterium_low)
    idx_high_dist = ds.getFeatIdx(dist_criterium_high)
    idx_rsi = ds.getFeatIdx(osc_criterium)    
    
    predictions = np.zeros (ds.X.shape[0])
    for i in range (len(predictions)):
        dist_low = np.abs(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0])
        dist_high = np.abs(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0])
        
        dist_21dma = ds.X[i,-1,idx_50dma] - 1.0
        dist_50dma = ds.X[i,-1,idx_50dma] - 1.0
        dist_100dma = ds.X[i,-1,idx_100dma] - 1.0
        dist_200dma = ds.X[i,-1,idx_200dma] - 1.0

        min_dist_l = np.min([np.abs(dist_100dma), np.abs(dist_200dma), dist_low])
        #min_dist_l = np.abs(dist_low)
        #min_dist_h = np.abs(dist_high)
        min_dist_h = np.min([np.abs(dist_100dma), np.abs(dist_200dma), dist_high])
        
        if (ds.X[i,-1,idx_ratio] > 1 + threshold) and \
                (ds.X[i,-1,idx_lines_ratio] > trending_ratio_threshold) and \
                    (min_dist_l<=dist_threshold) and \
                    (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold) and \
                     (ds.X[i,-1,idx_rsi] <= 70) and \
                     (np.abs(dist_21dma) <= 0.025):
            #print (str(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0]))
            predictions [i] = 2
        elif (ds.X[i,-1,idx_ratio] < 1 - threshold) and \
            (ds.X[i,-1,idx_lines_ratio] < (1.0/trending_ratio_threshold)) and \
            (min_dist_h<=dist_threshold) and \
            (ds.X[i,-1,idx_vol_ratio] >= vol_ratio_threshold) and \
            (ds.X[i,-1,idx_rsi] >= 30) and \
            (np.abs(dist_21dma) <= 0.025):
            predictions [i] = 0
            #print (str(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0]))
        else:
            predictions [i] = 1
    return predictions, evaluate_rule (predictions, ds.y, verbose, plotPred)
    
def rule_complexRSIFading (ds, threshold=0.01, criterium='close_over_50d_ma', 
                                      trending_criterium='ratio_standing_up_downward_lines_10', 
                                      trending_ratio_threshold=0.25,
                                      dist_criterium_low='dist_relevant_low_30',
                                      dist_criterium_high='dist_relevant_high_30',
                                      dist_threshold=0.015,                                      
                                      vol_criterium='2w_over_3m_vol_ratio',
                                      vol_ratio_threshold=0.98,
                                      osc_criterium='RSI',
                                      osc_threshold_high=65,
                                      osc_threshold_low=35,
                                      lookback_window=22,
                                      min_osc_dist = 10.0,
                                      min_dist_low_high = -0.025,
                                      max_dist_low_high = 0.10,
                                      verbose=False,
                                      plotPred=False):
        if verbose:
            print ('Simple RSI fading Rule')
    
        predictions = np.zeros (ds.X.shape[0])        
        
        idx_rsi_minus_peak = ds.getFeatIdx(feat='RSI_close_minus_peak', 
                      func=feat_metrics, 
                      args={'lookback_window':lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_peak'})
        
        idx_rsi_minus_bottom = ds.getFeatIdx(feat='RSI_close_minus_bottom', 
                      func=feat_metrics, 
                      args={'lookback_window':lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_bottom'})
        
        idx_criterium = ds.getFeatIdx(criterium)
        idx_rsi = ds.getFeatIdx(osc_criterium)
        
        idx_lines_ratio = ds.getFeatIdx(trending_criterium)
        idx_low_dist = ds.getFeatIdx(dist_criterium_low)
        idx_high_dist = ds.getFeatIdx(dist_criterium_high)
        
        idx_vol_ratio = ds.getFeatIdx(vol_criterium)
        threshold = osc_threshold_high
        
        for i in range (len(predictions)):
            dist_low = np.abs(ds.X[i,-1,idx_low_dist] / ds.X[i,-1,0])
            dist_high = np.abs(ds.X[i,-1,idx_high_dist] / ds.X[i,-1,0])
            
            if (ds.X[i,-1,idx_rsi] >= osc_threshold_high) and \
                        (ds.X[i,-1,idx_rsi_minus_peak] <= -min_osc_dist) and \
                        (dist_low >= min_dist_low_high) and \
                        (dist_high <= max_dist_low_high) and \
                        (ds.X[i,-1,idx_lines_ratio] >= trending_ratio_threshold) and \
                        (ds.X[i,-1,idx_vol_ratio] <= vol_ratio_threshold):
                #print ('short: i='+str(i)+', '+str(ds.X[i,-1,idx_ratio]))
                predictions [i] = 0
            elif (ds.X[i,-1,idx_rsi] <= (100 - osc_threshold_high)) and \
                    (ds.X[i,-1,idx_rsi_minus_bottom] >= min_osc_dist) and \
                    (dist_high >= min_dist_low_high) and \
                    (dist_low <= max_dist_low_high) and \
                    (ds.X[i,-1,idx_lines_ratio] <= 1.0 / trending_ratio_threshold) and \
                    (ds.X[i,-1,idx_vol_ratio] <= vol_ratio_threshold):
                predictions [i] = 2
                #print ('long: i='+str(i)+', '+str(ds.X[i,-1,idx_ratio]))
            else:
                predictions [i] = 1
        return predictions, evaluate_rule (predictions, ds.y, verbose, plotPred)

#----------Multi timeframe rules---------------------------------------#
#Checks if close is above 200dma (by at least 2.5%) on a daily basis
#Then looks for oversold/overbought conditions on the 4h timeframe
#Then buys/sells whenever RSI crosses above/below 33/67
def rule_mtf_simpleMomentum (ds_holder=None, args = {},
                             criterium_d='RSI',
                             criterium_d_threshold_high=50,
                             criterium_d_threshold_low=50,
                             criterium_h4='RSI',
                             criterium_h4_threshold_high=70,
                             criterium_h4_threshold_low=30,
                             dist_fast_osc_extrema = 0.0,
                             lookback_window=22,
                             pred_lag = 1,
                             pred_lag_d = 1,
                             verbose=False):
    if 'pred_lag' in args:
        pred_lag = args['pred_lag']

    if 'criterium_d_threshold_high' in args:
        criterium_d_threshold_high = args['criterium_d_threshold_high']
    if 'criterium_d_threshold_low' in args:
        criterium_d_threshold_low = args['criterium_d_threshold_low']
    
    if 'criterium_h4_threshold_low' in args:
        criterium_h4_threshold_low = args['criterium_h4_threshold_low']
    if 'criterium_h4_threshold_high' in args:
        criterium_h4_threshold_high = args['criterium_h4_threshold_high']

    if 'dist_fast_osc_extrema' in args:
        dist_fast_osc_extrema = args['dist_fast_osc_extrema']

    print ('Prediction lag (to avoid looking into future data: ' + str (pred_lag))
    
    keys = list(ds_holder.ds_dict.keys ())    
    ccy_pair = ds_holder.ds_dict[keys[0]].ccy_pair

    ds_d = ds_holder.ds_dict[ccy_pair+'_D']
    ds_h4 = ds_holder.ds_dict[ccy_pair+'_H4']

    idx_criterium_d = ds_d.getFeatIdx (criterium_d)
    
    idx_criterium_h4 = ds_h4.getFeatIdx (criterium_h4)

    idx_rsi_minus_peak = ds_h4.getFeatIdx(feat='RSI_close_minus_peak', 
                      func=feat_metrics, 
                      args={'lookback_window':lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_peak'})

    idx_rsi_minus_bottom = ds_h4.getFeatIdx(feat='RSI_close_minus_bottom', 
                      func=feat_metrics, 
                      args={'lookback_window':lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_bottom'})

    predictions = np.zeros (ds_h4.X.shape[0])
    for i in range (pred_lag, len(predictions)):
        predictions[i] = NEUTRAL_SIGNAL
        try:
        #if True:
            ts = ds_h4.f_df.index[i] - relativedelta(days=pred_lag_d)
            ts2 = ts - relativedelta(days=5)
            row_d = ds_d.f_df.loc[ts2:ts].ix[-1,:]

            if np.mod(i, 100) == 0:
                #print (str(ts) +', ' + str(ds_d.f_df.loc[ts2:ts].index[-1]))
                pass

            if ts < ds_d.f_df.loc[ts2:ts].index[-1]:                
                print ('Look ahead fault h4:' + str (ts) + 'd: ' + str(ds_d.f_df.loc[ts2:ts].index[-1]))
            
            if row_d[idx_criterium_d] > criterium_d_threshold_high and \
                    ds_h4.f_df.ix[i - pred_lag, idx_criterium_h4] < criterium_h4_threshold_low and \
                    ds_h4.X[i - pred_lag, -1, idx_rsi_minus_bottom] > dist_fast_osc_extrema:
                predictions[i] = LONG_SIGNAL   
            elif row_d[idx_criterium_d] < criterium_d_threshold_low and \
                    ds_h4.f_df.ix[i - pred_lag, idx_criterium_h4] > criterium_h4_threshold_high and \
                    ds_h4.X[i - pred_lag, -1, idx_rsi_minus_peak] < -dist_fast_osc_extrema:
                predictions[i] = SHORT_SIGNAL
        except:
            if verbose:
                print ('Error on prediction: ' + str(i))
            pass
    try:
        pass
        #print (str(ts) +', ' + str(ds_d.f_df.loc[ts2:ts].index[-1]))
    except:
        pass
    return predictions
    
def rule_mtf_complexRule (ds_holder = None, args ={},
                                     threshold=0.01, criterium='close_over_100d_ma', 
                                      trending_criterium='ratio_standing_up_downward_lines_10', 
                                      trending_ratio_threshold=2.0,
                                      dist_criterium_low='dist_relevant_low_30',
                                      dist_criterium_high='dist_relevant_high_30',
                                      dist_threshold=0.015,                                      
                                      vol_criterium='2w_over_3m_vol_ratio',
                                      vol_ratio_threshold=0.90,
                                      osc_criterium='RSI',
                                      osc_threshold_high=70,
                                      osc_threshold_low=30,
                                      criterium_h4='RSI',
                                      lookback_window=22,
                                      verbose=False,
                                      plotPred=False):
    if verbose:        
        print ('complex rule multi-timeframe')
        
    keys = list(ds_holder.ds_dict.keys ())    
    ccy_pair = ds_holder.ds_dict[keys[0]].ccy_pair

    ds_d = ds_holder.ds_dict[ccy_pair+'_D']
    ds_h4 = ds_holder.ds_dict[ccy_pair+'_H4']
    
    predictions = np.zeros (ds_h4.X.shape[0])
    idx_ratio = ds_d.getFeatIdx(criterium)
    idx_vol_ratio = ds_d.getFeatIdx(vol_criterium)
    idx_lines_ratio = ds_d.getFeatIdx(trending_criterium)
    
    idx_low_dist = ds_d.getFeatIdx(dist_criterium_low)
    idx_high_dist = ds_d.getFeatIdx(dist_criterium_high)
    
    idx_criterium_h4 = ds_h4.getFeatIdx (criterium_h4)    
    idx_rsi_minus_peak = ds_h4.getFeatIdx(feat='RSI_close_minus_peak', 
                      func=feat_metrics, 
                      args={'lookback_window':lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_peak'})
    idx_rsi_minus_bottom = ds_h4.getFeatIdx(feat='RSI_close_minus_bottom', 
                      func=feat_metrics, 
                      args={'lookback_window':lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_bottom'})
    
    for i in range (len(predictions)):        
        #return predictions, evaluate_rule (predictions, ds.y, verbose, plotPred)
        predictions[i] = 1
        #try:
        if True:
            ts = ds_h4.f_df.index[i]
            ts2 = ts - relativedelta(days=5)
            row_d = ds_d.f_df.loc[ts2:ts].ix[-1,:]

            if (row_d[idx_ratio] > 1 + threshold) and \
                (row_d[idx_lines_ratio] > trending_ratio_threshold) and \
                        (row_d[idx_vol_ratio] >= vol_ratio_threshold):
            #(np.abs(row_d[idx_low_dist] / row_d[0] - 1.0)<=dist_threshold) or \                    
                if ds_h4.f_df.ix[i, idx_criterium_h4] < 33.0 and \
                        ds_h4.X[i, -1, idx_rsi_minus_bottom] > 5.0:
                            predictions[i] = 2
            elif (row_d[idx_ratio] < 1 - threshold) and \
                (row_d[idx_lines_ratio] < (1.0/trending_ratio_threshold)) and \
                (row_d[idx_vol_ratio] >= vol_ratio_threshold):    
            #(np.abs(row_d[idx_high_dist] / row_d[0] - 1.0)<=dist_threshold) or \                
                if ds_h4.f_df.ix[i, idx_criterium_h4] > 67.0 and \
                        ds_h4.X[i, -1, idx_rsi_minus_peak] < -5.0:
                            
                    predictions[i] = 0
        #except:
        #    print ('Error on prediction - ' + str (i))
    
    return predictions
    
#should decide whether to chase or not
def rule_mtf_crossoverMomentumRule (ds_holder = None, args = {},                                     
                                      trending_criterium='ratio_standing_up_downward_lines_10', #ok
                                      trending_ratio_threshold_high=0.5,         #ok
                                      trending_ratio_threshold_low=2.0,
                                      #dist_criterium_low='dist_relevant_low_30',
                                      #dist_criterium_high='dist_relevant_high_30',
                                      crossover_fast = 'ma_50_close',   #ok
                                      crossover_slow = 'ma_200_close',  #ok                                                                          
                                      #vol_ratio_threshold=0.90,
                                      osc_criterium_slow='RSI',      #ok
                                      osc_threshold_slow_high=70,    #ok
                                      osc_threshold_slow_low=30,     #ok
                                      osc_criterium_fast='RSI',     #ok
                                      osc_threshold_fast_high=55,   #ok
                                      osc_threshold_fast_low=45,    #ok
                                      step_width=22,            #ok
                                      pred_lag_d = 1,
                                      verbose=False,
                                      plotPred=False):
    if verbose:        
        print ('mtf cross-over momentum rule')
       
    if 'trending_criterium' in args:
        trending_criterium = args['trending_criterium']

    if 'trending_ratio_threshold_high' in args:
        trending_ratio_threshold_high = args['trending_ratio_threshold_high']

    if 'trending_ratio_threshold_low' in args:
        trending_ratio_threshold_low = args['trending_ratio_threshold_low']

    if 'osc_criterium_slow' in args:
        osc_criterium_slow = args['osc_criterium_slow']

    if 'osc_threshold_slow_high' in args:
        osc_threshold_slow_high = args['osc_threshold_slow_high']

    if 'osc_threshold_slow_low' in args:
        osc_threshold_slow_low = args['osc_threshold_slow_low']

    if 'osc_criterium_fast' in args:
        osc_criterium_fast = args['osc_criterium_fast']

    if 'osc_threshold_fast_high' in args:
        osc_threshold_fast_high = args['osc_threshold_fast_high']

    if 'osc_threshold_fast_low' in args:
        osc_threshold_fast_low = args['osc_threshold_fast_low']    
    
    if 'crossover_fast' in args:        
        crossover_fast = args['crossover_fast']
        print ('Crossover fast in args: ' + str(crossover_fast))
        
    if 'crossover_slow' in args:
        crossover_slow = args['crossover_slow']
        print ('Crossover slow in args: ' + str(crossover_slow))

    print ('fast: '+ crossover_fast + ', slow: '+ crossover_slow)

    if 'step_width' in args:
        step_width = args['step_width']

    print (str(osc_criterium_slow) + ', ' + str (osc_threshold_slow_high)+ ', ' + str(osc_threshold_fast_high))
        
    keys = list(ds_holder.ds_dict.keys ())    
    ccy_pair = ds_holder.ds_dict[keys[0]].ccy_pair

    ds_d = ds_holder.ds_dict[ccy_pair+'_D']
    ds_h4 = ds_holder.ds_dict[ccy_pair+'_H4']
    
    predictions = np.zeros (ds_h4.X.shape[0])
    momentum_criterium = ds_d.getFeatIdx(osc_criterium_slow)    
    
    idx_lines_ratio = ds_d.getFeatIdx(trending_criterium)    
    idx_criterium_h4 = ds_h4.getFeatIdx (osc_criterium_fast)
    
    idx_crossover = ds_d.getFeatIdx(feat='crossover_' + str(crossover_fast) + '_' + str (crossover_slow), 
                      func=crossover, 
                      args={'step_width':step_width, 
                      'fast':crossover_fast,
                      'slow':crossover_slow,
                      'metric':'crossover_window'})
    
    
    for i in range (len(predictions)):                
        predictions[i] = NEUTRAL_SIGNAL
        try:
            ts = ds_h4.f_df.index[i] - relativedelta(days=pred_lag_d)
            ts2 = ts - relativedelta(days=5)
            row_d = ds_d.f_df.loc[ts2:ts].ix[-1,:]

            if ts < ds_d.f_df.loc[ts2:ts].index[-1]:                
                print ('Look ahead fault h4:' + str (ts) + 'd: ' + str(ds_d.f_df.loc[ts2:ts].index[-1]))
            else:
                if (row_d[idx_lines_ratio] > trending_ratio_threshold_high) and \
                            (row_d[idx_crossover] > 0) and (row_d[momentum_criterium] >= osc_threshold_slow_high):                               
                    if ds_h4.f_df.ix[i, idx_criterium_h4] <= osc_threshold_fast_low:
                                predictions[i] = LONG_SIGNAL
                elif (row_d[idx_lines_ratio] < trending_ratio_threshold_low) and \
                        (row_d[idx_crossover] < 0) and (row_d[momentum_criterium] <= osc_threshold_slow_low):                
                    if ds_h4.f_df.ix[i, idx_criterium_h4] >= osc_threshold_fast_high:                            
                        predictions[i] = SHORT_SIGNAL
        except:            
            if verbose: 
                print ('Error on prediction - ' + str (i))
            else:
                pass
    
    return predictions
    
############################################################################################################
    
def rule_mtf_chase_new_highs_lows (ds_holder = None, args = {},
                                      osc_criterium = 'RSI',
                                      osc_threshold_high = 60.0,
                                      osc_threshold_low = 40.0,
                                      osc_dist_extrema = 5.0,
                                      osc_lookback_window = 20,
                                      hi_lo_lookback_window = 252,
                                      step_width = 20,
                                      high_timeframe='D',
                                      low_timeframe='H4',
                                      verbose = False,
                                      plotPred = False):
    if verbose:        
        print ('complex rule multi-timeframe Chase Highs Momentum Rule')
        
    if 'osc_criterium' in args:
        osc_criterium = args['osc_criterium']
        
    if 'high_timeframe' in args:
        high_timeframe = args['high_timeframe']

    if 'low_timeframe' in args:
        low_timeframe = args['low_timeframe']

    if 'osc_threshold_high' in args:
        osc_threshold_high = args['osc_threshold_high']

    if 'osc_threshold_low' in args:
        osc_threshold_low = args['osc_threshold_low']

    if 'osc_dist_extrema' in args:
        osc_dist_extrema = args['osc_dist_extrema']

    if 'osc_lookback_window' in args:
        osc_lookback_window = args['osc_lookback_window']

    if 'hi_lo_lookback_window' in args:
        hi_lo_lookback_window = args['hi_lo_lookback_window']

    if 'step_width' in args:
        step_width = args['step_width']

    print (str(osc_threshold_low) + ', ' + str (osc_dist_extrema)+ ', ' + str(osc_lookback_window))
    
        
    keys = list(ds_holder.ds_dict.keys ())    
    ccy_pair = ds_holder.ds_dict[keys[0]].ccy_pair

    ds_d = ds_holder.ds_dict[ccy_pair + '_' + high_timeframe]
    ds_h4 = ds_holder.ds_dict[ccy_pair + '_' + low_timeframe]
    
    predictions = np.zeros (ds_h4.X.shape[0])
    
    idx_criterium_h4 = ds_h4.getFeatIdx (osc_criterium)
    
    idx_rsi_minus_peak = ds_h4.getFeatIdx(feat='RSI_close_minus_peak', 
                      func=feat_metrics, 
                      args={'lookback_window':osc_lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_peak'})
    idx_rsi_minus_bottom = ds_h4.getFeatIdx(feat='RSI_close_minus_bottom', 
                      func=feat_metrics, 
                      args={'lookback_window':osc_lookback_window, 
                      'feat':'RSI', 
                      'metric':'close_minus_bottom'})
    
    idx_new_hilo = ds_d.getFeatIdx(feat='new_hilo_' + str(hi_lo_lookback_window) + '_' + str (step_width), 
                      func=new_high_over_lookback_window, 
                      args={'lookback_window':hi_lo_lookback_window, 
                      'step_width':step_width,                      
                      'feat':'Close'})
    
    
    for i in range (len(predictions)):                
        predictions[i] = NEUTRAL_SIGNAL
        try:
            ts = ds_h4.f_df.index[i]
            ts2 = ts - relativedelta(days=5)
            row_d = ds_d.f_df.loc[ts2:ts].ix[-1,:]

            if row_d[idx_new_hilo] > 0.0:
                if verbose:
                    print ('Cleared daily criterium to buy: ')
                    print (str(i) + ', ' + str(ds_h4.f_df.ix[i, idx_criterium_h4]) + ', ' + str(ds_h4.X[i, -1, idx_rsi_minus_bottom]) )
                if ds_h4.f_df.ix[i, idx_criterium_h4] < osc_threshold_low and \
                        ds_h4.X[i, -1, idx_rsi_minus_bottom] > osc_dist_extrema:
                            predictions[i] = LONG_SIGNAL
            elif row_d[idx_new_hilo] < 0.0:
                if verbose:
                    print('Cleared daily criterium to sell: ')
                    print (str(i) + ', ' + str(ds_h4.f_df.ix[i, idx_criterium_h4]) + ', ' + str(ds_h4.X[i, -1, idx_rsi_minus_bottom]) )
                if ds_h4.f_df.ix[i, idx_criterium_h4] > osc_threshold_high and \
                        ds_h4.X[i, -1, idx_rsi_minus_peak] < -osc_dist_extrema:
                            predictions[i] = SHORT_SIGNAL
        except:            
            if verbose: 
                print ('Error on prediction - ' + str (i))
            else:
                pass
    
    return predictions


    
############################################################################################################
            
if False:
    ds = Dataset(lookback_window=2, n_features=239)
    ds.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume/NewFeatures'
    ds.feat_filename_prefix = 'not_normalized_new_feat_'
    ds.labelpath = './datasets/Fx/Labeled/Symmetrical'
    ds.label_filename_prefix = 'ccy_hist_label_'
    #ds.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume'
    ds.period_ahead = 3
    ds.last=500
    ds.cv_set_size = 1150
    ds.test_set_size = 1000
    
    series_no = 16
    
    series_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,40,41,42,43,44,46,47,48,49,50,
                 51,52,53, 54, 55, 56, 57, 58, 59, 61, 63, 64, 65, 66,  68, 69, 70, 
                 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 98, 99,
                 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
                 
    rules_func_list = [rule_SimpleMomentum, 
                       rule_SimpleMomentumWithTrendingFilter,
                       rule_SimpleMomentumWithVIXFilter,
                       rule_SimpleMomentumWithVolFilter,
                       rule_ComplexMomentumWithTrendingFilter,
                       rule_complexRule,
                       rule_complexRuleWithRSI,
                       rule_veryComplex]
                      
    #rules_func_list = [rule_complexRSIFading]
    
    acc_list = []
    neutral_list = []
    for series_no in series_list[20:]:
        try:
            ds.loadSeriesByNo (series_no, bRelabel=False, bNormalize=False, bConvolveCdl=True)
            #ds.loadDataSet(series_list=np.random.shuffle(series_list), end=15, bRelabel=False, bNormalize=False)
            #ds.createSingleTrainSet(3)
        
            if True:
                print ('------------------------------------------------------------------------')
                
                
                
                if False:
                    ds.y = relabel_custom(ds.X, ds.y)
                    ds.cv_y = relabel_custom(ds.cv_X, ds.cv_y)
                    ds.test_y = relabel_custom(ds.test_X, ds.test_y)
                
                acc_elem_list = []
                neutral_elem_list = []
                for func in rules_func_list:
                    pred, [acc, neutral_ratio] = func (ds, verbose=True)
                    acc_elem_list.append (acc)
                    neutral_elem_list.append (neutral_ratio)
                acc_list.append (acc_elem_list)
                neutral_list.append (neutral_elem_list)
            
    
        
           
        except:
            print ('Error with series '+str(series_no))
            
    acc_array = np.array (acc_list)
    neutral_array = np.array(neutral_list)
            
    np.nanmean((acc_array-0.5) * (1.0-neutral_array), axis=0) / np.nanmean(1.0-neutral_array, axis=0)
        
