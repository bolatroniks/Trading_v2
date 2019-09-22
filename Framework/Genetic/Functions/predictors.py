# -*- coding: utf-8 -*-

#dshdushd
import numpy as np

from Config.const_and_paths import NEUTRAL_SIGNAL, LONG_SIGNAL, SHORT_SIGNAL
from Miscellaneous.my_utils import parse_kwargs
from Framework.Genetic.Functions.threshold_inverters import *

predictor_functions = {}

#generic function to compute prevent predictions
def fn_pred_preventer (ds, **kwargs):
    indic = parse_kwargs (['indic', 'indicator'], 'RSI_overbought', **kwargs)
    dual_indic = parse_kwargs (['dual', 'dual_indic', 'dual_indicator'], None, **kwargs)
    
    #threshold_min = parse_kwargs (['threshold_min'], -0.5, **kwargs)
    threshold_max = parse_kwargs (['threshold_max'], 0.5, **kwargs)
    threshold_min = parse_kwargs (['threshold_min'], -0.5, **kwargs)
    
    inv_threshold_fn = parse_kwargs (['inv_threshold_fn', 'inv_fn'], inv_fn_symmetric, **kwargs)
    
    if dual_indic is not None:
        ret_prevent_buy = (ds.f_df[indic] > threshold_max) | (ds.f_df[indic] < threshold_min)
        ret_prevent_sell = (ds.f_df[dual_indic] > threshold_max) | (ds.f_df[dual_indic] < threshold_min);
    else:
        ret_prevent_buy = (ds.f_df[indic] > threshold_max) | (ds.f_df[indic] < threshold_min)
        ret_prevent_sell = (ds.f_df[indic] < inv_threshold_fn(threshold_max) ) | (ds.f_df[indic] > inv_threshold_fn(threshold_min))
    
    return ret_prevent_buy, ret_prevent_sell
    
def fn_pred_asymmetric (ds, **kwargs):
    indic = parse_kwargs (['indic', 'indicator'], 'RSI', **kwargs)
    
    threshold_min = parse_kwargs (['threshold_min'], -9999, **kwargs)
    threshold_max = parse_kwargs (['threshold_max'], 9999, **kwargs)

    direction = parse_kwargs (['direction', 'direc', 'long_short', 'buy_sell'], 1, **kwargs)


    ret = np.ones (len(ds.f_df)) * NEUTRAL_SIGNAL
    ret [(ds.f_df[indic] > threshold_min) & (ds.f_df[indic] < threshold_max)] = LONG_SIGNAL * direction

    return ret
    

#generic function to compute predictions based on one feature
def fn_pred3 (ds, **kwargs):
    indic = parse_kwargs (['indic', 'indicator'], 'RSI', **kwargs)
    dual_indic = parse_kwargs (['dual', 'dual_indic', 'dual_indicator'], None, **kwargs)
    threshold_min = parse_kwargs (['threshold_min'], -9999, **kwargs)
    threshold_max = parse_kwargs (['threshold_max'], 9999, **kwargs)

    inv_threshold_fn = parse_kwargs (['inv_threshold_fn', 'inv_fn'], None, **kwargs)


    ret = np.ones (len(ds.f_df)) * NEUTRAL_SIGNAL
    ret [(ds.f_df[indic] > threshold_min) & (ds.f_df[indic] < threshold_max)] = LONG_SIGNAL
    if inv_threshold_fn is not None:
        ret [(ds.f_df[(indic if dual_indic is None else dual_indic)] < inv_threshold_fn(threshold_min)) & (ds.f_df[(indic if dual_indic is None else dual_indic)] > inv_threshold_fn(threshold_max))] = SHORT_SIGNAL

    return ret

predictor_functions ['simple'] = {'func':fn_pred3, 
                                    'kwargs':{'indicator':'RSI',
                                              'dual_indicator': None,
                                              'threshold_min': -9999,
                                              'threshold_max': 9999,
                                              'inv_threshold_fn': inv_fn_rsi
                                                  }}

#generic function to compute predictions based on two features
def fn_pred_double (ds, **kwargs):
    indic1 = parse_kwargs (['indic1', 'indicator1'], 'RSI', **kwargs)
    dual_indic1 = parse_kwargs (['dual1', 'dual_indic1', 'dual_indicator'], None, **kwargs)
    threshold_min1 = parse_kwargs (['threshold_min1'], -9999, **kwargs)
    threshold_max1 = parse_kwargs (['threshold_max1'], 9999, **kwargs)
    inv_threshold_fn1 = parse_kwargs (['inv_threshold_fn1', 'inv_fn'], None, **kwargs)

    indic2 = parse_kwargs (['indic2', 'indicator2'], 'RSI', **kwargs)
    dual_indic2 = parse_kwargs (['dual2', 'dual_indic2', 'dual_indicator'], None, **kwargs)
    threshold_min2 = parse_kwargs (['threshold_min2'], -9999, **kwargs)
    threshold_max2 = parse_kwargs (['threshold_max2'], 9999, **kwargs)
    inv_threshold_fn2 = parse_kwargs (['inv_threshold_fn2', 'inv_fn'], None, **kwargs)

    ret = np.ones (len(ds.f_df)) * NEUTRAL_SIGNAL
    ret [(ds.f_df[indic1] > threshold_min1) & \
         (ds.f_df[indic1] < threshold_max1) & \
         (ds.f_df[indic2] > threshold_min2) & \
         (ds.f_df[indic2] < threshold_max2)] = LONG_SIGNAL

    if inv_threshold_fn1 is not None and inv_threshold_fn2 is None:
        ret [(ds.f_df[(indic1 if dual_indic1 is None else dual_indic1)] < inv_threshold_fn1(threshold_min1)) & \
             (ds.f_df[(indic1 if dual_indic1 is None else dual_indic1)] > inv_threshold_fn1(threshold_max1)) & \
             (ds.f_df[(indic2 if dual_indic2 is None else dual_indic2)] < threshold_min2) & \
             (ds.f_df[(indic2 if dual_indic2 is None else dual_indic2)] > threshold_max2)
             ] = SHORT_SIGNAL

    elif inv_threshold_fn1 is None and inv_threshold_fn2 is not None:
        ret [(ds.f_df[(indic1 if dual_indic1 is None else dual_indic1)] < threshold_min1) & \
             (ds.f_df[(indic1 if dual_indic1 is None else dual_indic1)] > threshold_max1) & \
             (ds.f_df[(indic2 if dual_indic2 is None else dual_indic2)] < inv_threshold_fn2(threshold_min2)) & \
             (ds.f_df[(indic2 if dual_indic2 is None else dual_indic2)] > inv_threshold_fn2(threshold_max2))
             ] = SHORT_SIGNAL

    elif inv_threshold_fn1 is not None and inv_threshold_fn2 is not None:
        ret [(ds.f_df[(indic1 if dual_indic1 is None else dual_indic1)] < inv_threshold_fn1(threshold_min1)) & \
             (ds.f_df[(indic1 if dual_indic1 is None else dual_indic1)] > inv_threshold_fn1(threshold_max1)) & \
             (ds.f_df[(indic2 if dual_indic2 is None else dual_indic2)] < inv_threshold_fn2(threshold_min2)) & \
             (ds.f_df[(indic2 if dual_indic2 is None else dual_indic2)] > inv_threshold_fn2(threshold_max2))
             ] = SHORT_SIGNAL
    return ret

predictor_functions ['double'] = {'func':fn_pred_double, 
                                    'kwargs':{'indicator1':'RSI',
                                              'dual_indicator1': None,
                                              'threshold_min1': -9999,
                                              'threshold_max1': 9999,
                                              'inv_threshold_fn1': inv_fn_rsi,
                                              'indicator2':'RSI',
                                              'dual_indicator2': None,
                                              'threshold_min2': -9999,
                                              'threshold_max2': 9999,
                                              'inv_threshold_fn2': inv_fn_rsi
                                                  }}