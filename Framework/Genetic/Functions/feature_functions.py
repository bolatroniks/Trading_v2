# -*- coding: utf-8 -*-

#dsdsd
import numpy as np

from Config.const_and_paths import LONG_SIGNAL, SHORT_SIGNAL
from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.DatasetHolder import DatasetHolder
from Framework.Dataset.dataset_func import get_from_time
from Framework.Features.TimeSeries.ernst_chen import halflife
from Miscellaneous.my_utils import parse_kwargs

from Framework.Dataset.RealTime.Realtime import compute_candle_features_on_the_fly

feature_functions = {}

#lambda to compute predictions
indic = 'RSI';
threshold = 60;
fn_pred = lambda ds: (ds.f_df[indic] > threshold) * LONG_SIGNAL + (ds.f_df[indic] < 100 - threshold) * SHORT_SIGNAL
fn_pred2 = lambda ds: ds.f_df['new_hilo']
window = 126
fn_feat = lambda ds: (ds.f_df.Close >= ds.f_df.Close.rolling(window).max ()) * LONG_SIGNAL + \
                (ds.f_df.Close <= ds.f_df.Close.rolling(window).min ()) * SHORT_SIGNAL
                

def fn_over_bought_sold (ds, **kwargs):
    indic = parse_kwargs (['indic'], 'RSI', **kwargs)
    conv_window = parse_kwargs (['conv_window'], 20, **kwargs)
    threshold_overbought = parse_kwargs (['threshold_overbought'], 70, **kwargs)
    threshold_oversold = parse_kwargs (['threshold_oversold'], 30, **kwargs)
    
    ret_overbought = ds.f_df[indic] > threshold_overbought
    ret_overbought = np.convolve(ret_overbought, np.ones (conv_window))[0:len(ret_overbought)]
    
    ret_oversold = ds.f_df[indic] < threshold_oversold
    ret_oversold = np.convolve(ret_oversold, np.ones (conv_window))[0:len(ret_oversold)]
    
    ds.f_df[indic + '_overbought'] = ret_overbought
    ds.f_df[indic + '_oversold'] = ret_oversold
    
    return ret_overbought

def fn_new_hilo (ds, **kwargs):
    window = parse_kwargs (['window'], 126, **kwargs)
    conv_window = parse_kwargs (['conv_window'], 20, **kwargs)

    ret = (ds.f_df.Close >= ds.f_df.Close.rolling(window).max ()) * LONG_SIGNAL + \
                (ds.f_df.Close <= ds.f_df.Close.rolling(window).min ()) * SHORT_SIGNAL

    ret = np.convolve(ret, np.ones (conv_window))[0:len(ret)]
    
    ret = np.minimum(np.maximum(ret, -1), 1)

    return ret

feature_functions ['fn_new_hilo'] = {'func':fn_new_hilo, 
                                      'kwargs':{'window':126}}

def halflife_wrap (ds, **kwargs):
    window = parse_kwargs (['window'], 252, **kwargs)

    b = np.array([halflife (ds.f_df.Close[i-window:i]) for i in range(window, len(ds.f_df))])
    ds.f_df['halflife'] = np.zeros(len(ds.f_df))
    ds.f_df['halflife'][window:] = b[:,1]
    
    ds.f_df['halflife_lambda'] = np.zeros(len(ds.f_df))
    ds.f_df['halflife_lambda'][window:] = b[:,0]
    
    return ds.f_df['halflife']

feature_functions ['halflife_wrap'] = {'func':halflife_wrap, 
                                      'kwargs':{'window':252}}

def feat_off_low_high (ds, **kwargs):
    feat = parse_kwargs (['feat', 'feature', 'feat_name'], None, **kwargs)
    window = parse_kwargs (['window', 'lookback_window'], None, **kwargs)
    high_low = parse_kwargs (['high_low', 'min_max', 'extreme'], None, **kwargs)

    if high_low == 'low' or high_low == 'min':
        return (ds.f_df[feat] - ds.f_df[feat].rolling(window=window).min())
    else:
        return (ds.f_df[feat] - ds.f_df[feat].rolling(window=window).max())
    
feature_functions ['feat_off_low_high'] = {'func':feat_off_low_high, 
                                      'kwargs':{'feat':'RSI',
                                                'window':60,
                                                'high_low': 'high'}}

#computes a simple operation with two feats like difference, ratio, log_ratio
def feats_operation (ds, **kwargs):
    feat1 = parse_kwargs (['feat1', 'feature1'], None, **kwargs)
    feat2 = parse_kwargs (['feat2', 'feature2'], None, **kwargs)
    operation = parse_kwargs (['operation'], 'difference', **kwargs)
    denominator = parse_kwargs (['den', 'denominator', 'normalize_by', 'normalized_by'], 1.0, **kwargs)

    if type(denominator) == str:
        if denominator in ds.f_df.columns:
            denominator = ds.f_df [denominator]
        else:
            return None
    if feat1 is not None and feat2 is not None:
        if operation == 'difference':
            return ds.f_df[feat1] - ds.f_df[feat2] / denominator
        if operation == 'ratio':
            return float(ds.f_df[feat1]) / float(ds.f_df[feat2])
        if operation == 'log_ratio':
            return np.log(float(ds.f_df[feat1]) / float(ds.f_df[feat2]))
    return None

feature_functions ['feats_operation'] = {'func':feats_operation, 
                                      'kwargs':{'feat1':'RSI',
                                                'feat2':'RSI',
                                                'operation':'difference',
                                                'denominator': None}}

#removes all unnecessary features from dataset
def slim_dataset (ds, **kwargs):
    if 'feat_to_keep' in kwargs.keys ():
        feat_to_keep = kwargs ['feat_to_keep']

        for col in ds.f_df.columns:
            if (col not in feat_to_keep):
                del ds.f_df[col]

    return np.zeros (len(ds.f_df))

#to be used to save memory when the need might arise

#function that merges two datasets with two different timeframes
def merge_timeframes (ds, **kwargs):
    if 'ds_slow' in kwargs.keys ():
        ds_lower_tf = kwargs ['ds_slow']
    else:
        ds_lower_tf = None

    if 'slow_tf' in kwargs.keys ():
        slow_tf = kwargs ['slow_tf']
    else:
        if ds_lower_tf is not None:
            slow_tf = ds_lower_tf.timeframe
        slow_tf = 'D'

    if 'daily_delay' in kwargs.keys ():
        daily_delay = kwargs ['daily_delay']
    else:
        daily_delay = 1

    if 'bConvolveCdl' in kwargs.keys ():
        bConvolveCdl = kwargs ['bConvolveCdl']
    else:
        bConvolveCdl = True

    if ds_lower_tf is None:
        ds_lower_tf = Dataset(ccy_pair = ds.ccy_pair, timeframe = slow_tf,
                            from_time = get_from_time(ds.df.index[-1], slow_tf),
                            to_time=ds.df.index[-1],
                            bLoadCandlesOnline = ds.bLoadCandlesOnline)
    if ds_lower_tf.df is None:
        ds_lower_tf.loadCandles ()
    if ds_lower_tf.f_df is None:
        ds_lower_tf.computeFeatures (bComputeHighLowFeatures = False) #high low features are not calculated by default

    dsh = DatasetHolder(instrument = ds.ccy_pair,
                        from_time = ds.from_time, to_time = ds.to_time)
    dsh.ds_dict [ds.ccy_pair + '_' + ds.timeframe] = ds

    dsh.ds_dict [ds.ccy_pair + '_' + ds_lower_tf.timeframe] = ds_lower_tf

    dsh.appendTimeframesIntoOneDataset (higher_timeframe = ds_lower_tf.timeframe,
                                             lower_timeframe = ds.timeframe,
                                             daily_delay = daily_delay,
                                             bConvolveCdl = bConvolveCdl)

    return ds.f_df['Close_' + ds.timeframe]

#------------------------------calendar functions------------------------#
def is_first_business_day (ds, **kwargs):
    df = ds.f_df
    
    return [True] + list((np.diff([_.month for _ in df.index]) != 0))

def is_turn_of_month (ds, **kwargs):
    df = ds.f_df
    
    first_business_days = is_first_business_day (ds, **kwargs)
    
    days_ahead = parse_kwargs (['days_ahead'], 2, **kwargs)
    days_after = parse_kwargs (['days_after'], 1, **kwargs)
    
    ret = np.zeros (len (first_business_days))
    for i in range (len (first_business_days)):
        if True in list (first_business_days [np.maximum(i - days_ahead, 0):np.minimum(i + days_after + 1, len (first_business_days))]):
            ret [i] = 1
    return ret

def is_just_before_close (ds, **kwargs):
    df = ds.f_df
    
    periods_ahead = parse_kwargs (['periods_ahead'], 2, **kwargs)
    periods_after = parse_kwargs (['periods_after'], -1, **kwargs)
    closing_time = parse_kwargs (['closing_time'], 21, **kwargs)
    
    closings = [(_.hour == closing_time) & (_.minute == 0) for _ in df.index]
    
    ret = np.zeros (len (closings))
    for i in range (len (closings)):
        if True in list (closings [np.maximum(i - periods_ahead, 0):np.minimum(i + periods_after + 1, len (closings))]):
            ret [i] = 1
    return ret
    
#-------------------candle features-----------------------------------#
strong_bearish_reversals = ['CDL3BLACKCROWS', 
                           'CDLIDENTICAL3CROWS', 
                           'CDLEVENINGSTAR', 
                           'CDL3LINESTRIKE']

reliable_bearish_patterns = ['CDLEVENINGDOJISTAR',
                             'CDL3OUTSIDE', 
                             'CDLENGULFING',
                             'CDLBELTHOLD',
                             'CDLABANDONEDBABY'
                             ]

strong_bullish_reversals = ['CDL3WHITESOLDIERS',
                           'CDLMORNINGSTAR'
                           ]

reliable_bearish_patterns = ['CDL3LINESTRIKE',
                             'CDLMORNINGDOJISTAR',
                             'CDL3OUTSIDE', 
                             'CDLENGULFING',
                             'CDLBELTHOLD',
                             'CDLABANDONEDBABY'
                             ]

def fn_candle_reversal (ds, **kwargs):
    df = ds.f_df
    conv_window = parse_kwargs (['conv_window'], 20, **kwargs)
    
    if strong_bearish_reversals [0] not in df.columns:
        compute_candle_features_on_the_fly (ds.f_df)
    
    df['strong_bearish_reversals'] = SHORT_SIGNAL * np.convolve(df[[feat for feat in strong_bearish_reversals]].abs().sum(axis=1), np.ones (conv_window))[0:len(df)]
    df['strong_bullish_reversals'] = LONG_SIGNAL * np.convolve(df[[feat for feat in strong_bullish_reversals]].abs().sum(axis=1), np.ones (conv_window))[0:len(df)]

def fn_candle_features (ds, **kwargs):
    compute_candle_features_on_the_fly (ds.f_df)
    
def fn_hammer (ds, **kwargs):
    MIN_BAR_LENGTH = parse_kwargs (['MIN_BAR_LENGTH', 'min_bar_length'], 0.001, **kwargs)    
    MIN_CANDLE_BODY_RATIO = parse_kwargs (['MIN_CANDLE_BODY_RATIO', 'min_candle_body_ratio'], 2.5, **kwargs)
    conv_window = parse_kwargs (['conv_window'], 1, **kwargs)
    
    if 'CDLHAMMER' not in ds.f_df.columns:
        compute_candle_features_on_the_fly (ds.f_df)
        
    #compute candle
    signals = ((ds.f_df.CDLHAMMER == 1) & \
               (np.abs((ds.f_df.High - ds.f_df.Low) / (ds.f_df.Open - ds.f_df.Close) >= MIN_CANDLE_BODY_RATIO)) & \
                        (((ds.f_df.High - ds.f_df.Low) / ds.f_df.Open) >= MIN_BAR_LENGTH)) * LONG_SIGNAL
                
    signals += ((ds.f_df.CDLINVERTEDHAMMER == 1) & \
                (np.abs((ds.f_df.High - ds.f_df.Low) / (ds.f_df.Open - ds.f_df.Close) >= MIN_CANDLE_BODY_RATIO)) & \
                        (((ds.f_df.High - ds.f_df.Low) / ds.f_df.Open) >= MIN_BAR_LENGTH)) * SHORT_SIGNAL
    
    signals = np.convolve(signals, np.ones (conv_window))[0:len(signals)]
                
    return signals
