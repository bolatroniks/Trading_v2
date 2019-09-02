# -*- coding: utf-8 -*-

from Framework.Genetic.Gene import *
from Framework.Genetic.GenePCA import GenePCA
from Framework.Genetic.Chromossome import Chromossome

from Framework.Genetic.Functions.predictors import *

from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.DatasetHolder import DatasetHolder

from Config.const_and_paths import *
from Framework.Strategy.Utils.strategy_func import *

import numpy as np
import gc

#creates a dataset
ds = Dataset(ccy_pair='USD_ZAR', 
                              from_time = 2013,
                              to_time=2013, 
                              timeframe='H1')

#creates a chromossome
c_rsi_off_extreme = Chromossome (ds = ds, bDebug = True, bSlim = False)

#adds a gene
#RSI is a default feature, therefore it is not necessary to add a function in func_dict to compute it
#two new features are added:
#       - RSI_off_low: measures how far RSI is off the low observed in the last 20 periods;
#       - RSI_high_low: measures how far RSI is off the high observed in the last 20 periods.
#predictions are computed based on three indicators: 
#       - longs: 30 < RSI < 35 and 3.0 < RSI_off_low < 15.0;
#       - shorts: 65 < RSI < 70 and -15.0 < RSI_off_high < -3.0
c_rsi_off_extreme.add_gene(timeframe = 'H1',
                   func_dict = {
                            'RSI_off_low':{'func':feat_off_low_high, 
                                              'kwargs':{'feat': 'RSI',
                                                        'window': 20,
                                                        'high_low': 'low'}
                                              },
                            'RSI_off_high':{'func':feat_off_low_high, 
                                              'kwargs':{'feat': 'RSI',
                                                        'window': 20,
                                                        'high_low': 'high'}
                                              },
                                },
                     pred_label = 'RSI',
                     pred_func= fn_pred_double, 
                     pred_type = 'symmetric',   #TODO: check why this gene is defaulted to binary instead of symmetric
                     pred_kwargs = {'indic1': 'RSI',
                                         'threshold_min1': 30,
                                         'threshold_max1': 35,
                                         'inv_threshold_fn1': inv_fn_rsi,
                                         'indic2': 'RSI_off_low',
                                        'dual_indic2': 'RSI_off_high',
                                        'threshold_min2': 3.0,
                                        'threshold_max2': 15.0,
                                        'inv_threshold_fn2': inv_fn_symmetric #should be symmetric?
                                        })

c_rsi_off_extreme.run ()

#checks whether predictions computed manually match those of c_rsi_off_extreme.ds
ds = c_rsi_off_extreme.ds

longs = (ds.f_df.RSI_H1 > 30) & (ds.f_df.RSI_H1 < 35) & (ds.f_df.RSI_off_low_H1 > 3) & (ds.f_df.RSI_off_low_H1 < 15)
assert (c_rsi_off_extreme.ds.p_df['pred:RSI_H1'][longs].unique ()[0] == 1.0)
shorts = (ds.f_df.RSI_H1 > 65) & (ds.f_df.RSI_H1 < 70) & (ds.f_df.RSI_off_high_H1 < -3) & (ds.f_df.RSI_off_high_H1 > -15)
assert (c_rsi_off_extreme.ds.p_df['pred:RSI_H1'][shorts].unique ()[0] == -1.0)

#-----------------------preventer--------------------------------------#
c_preventer = deepcopy (c_rsi_off_extreme)


#add a symmetric preventer
c_preventer.add_gene(timeframe = 'D',
                     func_dict = {'dummy':{'func':fn_over_bought_sold, 
                                          'kwargs':{'conv_window': 10,
                                                    'threshold_overbought': 70,
                                                    'threshold_oversold': 30}
                                          }
                                            },
                     pred_label = 'RSI_prevent_overbought',
                     pred_func= fn_pred_preventer, 
                     pred_kwargs = {
                                    'pred_type': 'preventer',
                                    'indic': 'RSI_overbought',
                                    'dual_indic': 'RSI_oversold',
                                         'threshold_min': -0.5,
                                         'threshold_max': 0.5,
                                         'inv_threshold_fn': inv_fn_identity})

#adds a binary indicator
c_preventer.add_gene (timeframe = 'H4', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 0,
                                                'threshold_max': 25,
                                                'inv_threshold_fn': None
                                            })

#TODO: there is a bug here, the daily frame does not have candles loaded
#ds.df is None and ds.f_df has shape (0,XX)
c_preventer.run ()

print ('There is a problem in the predictions aggregation')
c_preventer.get_last_slow_timeframe_gene ().ds.f_df['RSI_overbought'].plot ()
c_preventer.ds.p_df['pred:RSI_H1'].plot ()
c_preventer.ds.p_df.Predictions.plot ()