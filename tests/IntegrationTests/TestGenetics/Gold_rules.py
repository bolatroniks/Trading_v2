# -*- coding: utf-8 -*-

from Framework.Genetic.Gene import *
from Framework.Genetic.GenePCA import GenePCA
from Framework.Genetic.Chromossome import Chromossome

from Framework.Genetic.Functions.predictors import *

from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.DatasetHolder import DatasetHolder

from Config.const_and_paths import *
from Framework.Strategy.Utils.strategy_func import *
from Framework.Strategy.StrategySimulation import *

import numpy as np
import gc


#works for gold, S&P and Nikkei
c = Chromossome (ds = Dataset(ccy_pair='XAU_USD',
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2000,
                              to_time=2013, 
                              timeframe='M15'), bDebug = True, bSlim = False)

c.add_gene (timeframe = 'D', func_dict = {'new_hilo':{'func':fn_new_hilo, 
                                                          'kwargs':{'window': 252,
                                                                    'conv_window': 25}
                                                          }
                                            },
                                pred_label = 'new_hilo',
                                pred_type = 'symmetric',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'new_hilo',
                                                'threshold_min': 0.5, 
                                                'threshold_max': 1.5,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })

#add a symmetric preventer
c.add_gene(timeframe = 'D',
                     func_dict = {'dummy':{'func':fn_over_bought_sold, 
                                          'kwargs':{'conv_window': 5,
                                                    'threshold_overbought': 73,
                                                    'threshold_oversold': 27}
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

c.add_gene(timeframe = 'W',
                     func_dict = {'dummy':{'func':fn_over_bought_sold, 
                                          'kwargs':{'conv_window': 2,
                                                    'threshold_overbought': 75,
                                                    'threshold_oversold': 25}
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

if False:
    c.add_gene(timeframe = 'M15',
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
if True:
    c.add_gene(timeframe = 'M15',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 25,
                                             'threshold_max': 35,
                                             'inv_threshold_fn': inv_fn_rsi})
    

c.run ()
c.ds.computeLabels (min_stop = 0.008, target_multiple = 1.5)
c.ds.removeSerialPredictions (25)
plot_pnl (c.ds)

#ToDo: refactor to look like this:
#kwargs = {
#            'update_stop': {'func': ..., kwargs: {...}},
#            'update_target': {'func': ..., kwargs: {...}},
#            'init_stop': {},
#            'init_target': {},
#            'trigger_entry':
#            'force_exit': 
#           }
kwargs = {'func_update_stop': fn_stop_update_trailing_v1,
            'func_trigger_entry' : None, #fn_stop_entry,
            'trailing_bars_trigger_entry' : 1,
            'func_init_stop': fn_stop_init_v1,
            'kill_after' : 3,
            'trailing_bars' : 100,
            'move_proportion' : 0.75}
strat = StrategySimulation (ds = c.ds, signals = None, **kwargs)
strat.run ()
strat.diagnostics ()