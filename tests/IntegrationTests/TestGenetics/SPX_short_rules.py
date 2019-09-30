# -*- coding: utf-8 -*-

from Framework.Genetic.Gene import *
from Framework.Genetic.GenePCA import GenePCA
from Framework.Genetic.Chromossome import Chromossome

from Framework.Genetic.Functions.predictors import *

from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.DatasetHolder import DatasetHolder

from Config.const_and_paths import *
from Framework.Strategy.Utils.strategy_func import *
from Framework.Strategy.StrategySimulation import StrategySimulation

import numpy as np
import gc

if True:
    ds = Dataset(ccy_pair='UK100_GBP', 
                              from_time = 2000,
                              to_time=2013, 
                              timeframe='H1')

    ds.loadCandles ()
    ds.computeFeatures (bComputeHighLowFeatures=False, bComputeNormalizedRatios=False, bComputeIndicators=True)
    ds.computeLabels (min_stop=0.015, periods_ahead_returns=[1,2,5])
    c = Chromossome (ds = ds, bDebug = True, bSlim = False)
    
    c.add_gene(timeframe = 'D',
                     func_dict = {'turn_month':{'func':is_turn_of_month, 
                                          'kwargs':{'days_ahead':3,
                                                            'days_after':2}
                                          }
                                            },
                     pred_label = 'turn_month_prevent',
                     pred_func = fn_pred_preventer, 
                     pred_type = 'preventer',
                     pred_kwargs = {
                                    'pred_type': 'preventer',
                                    'indic': 'turn_month',
                                    'dual_indic': 'turn_month',
                                     'threshold_min': -0.5,
                                     'threshold_max': 0.5,
                                     'inv_threshold_fn': inv_fn_identity})
    
    c.add_gene(timeframe = 'W',
                         pred_label = 'RSI_asymmetric',
                         pred_func= fn_pred_asymmetric, 
                         pred_type = 'asymmetric',   #TODO: check why this gene is defaulted to binary instead of symmetric
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 65,
                                             'threshold_max': 80,
                                             'direction': -1 #1 for long, -1 for short
                                            })
    
    
    c.add_gene(timeframe = 'D',
                         pred_label = 'RSI_asymmetric',
                         pred_func= fn_pred_asymmetric, 
                         pred_type = 'asymmetric',   #TODO: check why this gene is defaulted to binary instead of symmetric
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 30,
                                             'threshold_max': 50,
                                             'direction': -1 #1 for long, -1 for short
                                            })
    
    c.add_gene(timeframe = 'H1',
                             pred_label = 'RSI',
                             pred_func= fn_pred_asymmetric, 
                             pred_type = 'asymmetric',
                             pred_kwargs = {'indic': 'RSI',
                                                 'threshold_min': 65,
                                                 'threshold_max': 80,
                                                 'direction': -1
                                                 })
    
    c.run ()
    c.ds.removeSerialPredictions (10)
    plot_pnl (c.ds)

kwargs = {'func_update_stop': fn_stop_update_trailing_v1,
            'func_trigger_entry' : fn_stop_entry,
            'trailing_bars_trigger_entry' : 5,
            'kill_after' : 3,
            'trailing_bars' : 5,
            'move_proportion' : 0.7}
strat = StrategySimulation (ds = c.ds, signals = None, **kwargs)
strat.run ()