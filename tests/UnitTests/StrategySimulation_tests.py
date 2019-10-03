# -*- coding: utf-8 -*-

from Framework.Genetic.Functions.predictors import *

from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.DatasetHolder import DatasetHolder

from Config.const_and_paths import *
from Framework.Strategy.Utils.strategy_func import *
from Framework.Strategy.StrategySimulation import *

import numpy as np
import gc


#ToDo: test whether the two methods to pass kwargs to TradeSimulation yield similar results
#There are two ways to pass kwargs
#kwargs = {'init_stop':{...}}
#or kwargs = {'func_init_stop':..., 'func_init_target':..., etc}

#The former is preferrable

c = Chromossome (ds = Dataset(ccy_pair='AUD_USD',
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2000,
                              to_time=2013, 
                              timeframe='H1'), bDebug = True, bSlim = False)

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

c.add_gene(timeframe = 'H1',
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

c.run ()

kwargs = {'func_update_stop': fn_stop_update_trailing_v1,
          'func_init_target': fn_target_init_v1,
          'func_init_stop': fn_stop_init_v1,
          'func_force_exit': fn_force_exit_n_bars,
          'n_bars': 240,
            'func_trigger_entry' : None, #fn_stop_entry,
            'trailing_bars_trigger_entry' : 5,
            'kill_after' : 3,
            'trailing_bars' : 10,
            'target_multiple': 2,
            'move_proportion' : 0.5}
strat = StrategySimulation (ds = c.ds, signals = None, **kwargs)
strat.run ()
strat.diagnostics ()

kwargs2 = {
        'update_stop': {'func': fn_stop_update_trailing_v1, 
                        'kwargs': {
                                    'trailing_bars' : 10,
                                    'move_proportion' : 0.5,
                                }
                        },
        'init_target': {'func': fn_target_init_v1, 
                        'kwargs': {
                                    'trailing_bars' : 10,
                                    'target_multiple': 2,
                                }
                        },
        'init_stop': {'func': fn_stop_init_v1, 
                        'kwargs': {
                                    'trailing_bars' : 10,
                                    'move_proportion' : 0.5,
                                }
                        },
        'force_exit': {'func': fn_force_exit_n_bars, 
                        'kwargs': {
                                    'n_bars': 20,
                                }
                        }
        }
                        
strat2 = StrategySimulation (ds = c.ds, signals = None, **kwargs2)
strat2.run ()
strat2.diagnostics ()
