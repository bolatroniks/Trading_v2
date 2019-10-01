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

if False:
    c = Chromossome (ds = Dataset(ccy_pair='AUD_USD',
                                  #from_time='2015-10-01 00:00:00', 
                                  from_time = 2000,
                                  to_time=2013, 
                                  timeframe='D'), bDebug = True, bSlim = False)
    
    c.add_gene(timeframe = 'D',
                         func_dict = {'dummy':{'func':fn_candle_reversal, 
                                              'kwargs':{'conv_window': 20}
                                              }
                                                },
                         pred_label = 'reversal',
                         pred_type = 'symmetric',
                         pred_func= fn_pred3, 
                         pred_kwargs = {
                                        'pred_type': 'symmetric',
                                        'dual_indic': 'strong_bearish_reversals',
                                        'indic': 'strong_bullish_reversals',
                                             'threshold_min': 0.5,
                                             'threshold_max': 999.0,
                                             'inv_threshold_fn': inv_fn_symmetric})

if False:
    c = Chromossome (ds = Dataset(ccy_pair='XAU_USD',
                                  #from_time='2015-10-01 00:00:00', 
                                  from_time = 2000,
                                  to_time=2013, 
                                  timeframe='H4'), bDebug = True, bSlim = False)
    
    c.add_gene(timeframe = 'D',
                         func_dict = {'dummy':{'func':fn_candle_reversal, 
                                              'kwargs':{'conv_window': 20}
                                              }
                                                },
                         pred_label = 'reversal',
                         pred_type = 'symmetric',
                         pred_func= fn_pred3, 
                         pred_kwargs = {
                                        'pred_type': 'symmetric',
                                        'dual_indic': 'strong_bearish_reversals',
                                        'indic': 'strong_bullish_reversals',
                                             'threshold_min': 0.5,
                                             'threshold_max': 999.0,
                                             'inv_threshold_fn': inv_fn_symmetric})
    
    c.add_gene(timeframe = 'H4',
                                 pred_label = 'RSI',
                                 pred_func= fn_pred3, 
                                 pred_type = 'symmetric',
                                 pred_kwargs = {'indic': 'RSI',
                                                     'threshold_min': 51,
                                                     'threshold_max': 70,
                                                     'inv_threshold_fn': inv_fn_rsi})
    
    c.add_gene (timeframe = 'H4', func_dict = {'halflife':{'func':halflife_wrap, 
                                                              'kwargs':{'window': 375}
                                                              }
                                                },
                                    pred_label = 'halflife',
                                    pred_func = fn_pred3,
                                    pred_type = 'binary',
                                    pred_kwargs = {
                                                    'indic': 'halflife',
                                                    'threshold_min': 50, 
                                                    'threshold_max': 9990,
                                                    'inv_threshold_fn': None
                                                })

if True:
    c = Chromossome (ds = Dataset(ccy_pair='USD_ZAR',
                                  #from_time='2015-10-01 00:00:00', 
                                  from_time = 2000,
                                  to_time=2013, 
                                  timeframe='H1'), bDebug = True, bSlim = False)
    
    
    #ToDo: BUG - get_stats method ignores this gene
    c.add_gene(timeframe = 'D',
                         func_dict = {'dummy':{'func':fn_candle_reversal, 
                                              'kwargs':{'conv_window': 60}
                                              }
                                                },
                         pred_label = 'reversal',
                         pred_type = 'preventer',
                         pred_func= fn_pred_preventer, 
                         pred_kwargs = {
                                        'pred_type': 'preventer',
                                        'dual_indic': 'strong_bullish_reversals',
                                        'indic': 'strong_bearish_reversals',
                                             'threshold_min': -0.5,
                                             'threshold_max': 0.5,
                                             'inv_threshold_fn': inv_fn_symmetric})
        
    c.add_gene(timeframe = 'H1',
                                 pred_label = 'RSI',
                                 pred_func= fn_pred3, 
                                 pred_type = 'symmetric',
                                 pred_kwargs = {'indic': 'RSI',
                                                     'threshold_min': 55,
                                                     'threshold_max': 80,
                                                     'inv_threshold_fn': inv_fn_rsi})
    
    c.add_gene(timeframe = 'D',
                                 pred_label = 'RSI',
                                 pred_func= fn_pred3, 
                                 pred_type = 'symmetric',
                                 pred_kwargs = {'indic': 'RSI',
                                                     'threshold_min': 51,
                                                     'threshold_max': 70,
                                                     'inv_threshold_fn': inv_fn_rsi})
    
    c.add_gene(timeframe = 'W',
                                 pred_label = 'RSI',
                                 pred_func= fn_pred3, 
                                 pred_type = 'symmetric',
                                 pred_kwargs = {'indic': 'RSI',
                                                     'threshold_min': 51,
                                                     'threshold_max': 70,
                                                     'inv_threshold_fn': inv_fn_rsi})
    
    
    if False:
        c.add_gene(timeframe = 'D',
                                 pred_label = 'RSI',
                                 pred_func= fn_pred3, 
                                 
                                 pred_kwargs = {'indic': 'RSI',
                                                     'threshold_min': 51,
                                                     'threshold_max': 60,
                                                     'inv_threshold_fn': inv_fn_rsi})
        
        c.add_gene (timeframe = 'D', func_dict = {'new_hilo':{'func':fn_new_hilo, 
                                                              'kwargs':{'window': 500,
                                                                        'conv_window': 60}
                                                              }
                                                },
                                    pred_label = 'new_hilo',
                                    pred_type = 'symmetric',
                                    pred_func = fn_pred3,
                                    pred_kwargs = {
                                                    'indic': 'new_hilo',
                                                    'threshold_min': -9999.5, 
                                                    'threshold_max': -0.5,
                                                    'inv_threshold_fn': inv_fn_symmetric
                                                })
    
    c.add_gene (timeframe = 'H4', func_dict = {'halflife':{'func':halflife_wrap, 
                                                              'kwargs':{'window': 375}
                                                              }
                                                },
                                    pred_label = 'halflife',
                                    pred_func = fn_pred3,
                                    pred_type = 'binary',
                                    pred_kwargs = {
                                                    'indic': 'halflife',
                                                    'threshold_min': 50, 
                                                    'threshold_max': 9990,
                                                    'inv_threshold_fn': None
                                                })
    
    
c.run ()

c.ds.computeLabels ()
c.ds.computeLabels (min_stop = 0.02, target_multiple = 1.5)

c.ds.removeSerialPredictions (10)
plot_pnl (c.ds)

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