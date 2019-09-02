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



#Turn of the month strategy for the SPX
if True:
    ds = Dataset(ccy_pair='SPX500_USD', 
                              from_time = 2000,
                              to_time=2019, 
                              timeframe='H1')

    ds.loadCandles ()
    ds.computeFeatures (bComputeHighLowFeatures=False, bComputeNormalizedRatios=False, bComputeIndicators=True)
    ds.computeLabels (min_stop=0.015, periods_ahead_returns=[1,2,5])
    c = Chromossome (ds = ds, bDebug = True, bSlim = False)
    
    c.add_gene(timeframe = 'D',
                       func_dict = {
                                'turn_month':{'func':is_turn_of_month, 
                                                  'kwargs':{'days_ahead':2,
                                                            'days_after':0}
                                                  }                            
                                    },
                         pred_label = 'turn_month',
                         pred_func= fn_pred_asymmetric, 
                         pred_type = 'asymmetric',   #TODO: check why this gene is defaulted to binary instead of symmetric
                         pred_kwargs = {'indic': 'turn_month',
                                             'threshold_min': 0.5,
                                             'threshold_max': 1.5,
                                             'direction': 1 #1 for long, -1 for short
                                            })
    #if False:
    c.add_gene(timeframe = 'D',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 51,
                                             'threshold_max': 70,
                                             'inv_threshold_fn': inv_fn_rsi})    
    
    c.add_gene(timeframe = 'H1',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 25,
                                             'threshold_max': 40,
                                             'inv_threshold_fn': inv_fn_rsi})
    
    c.run ()
    c.ds.removeSerialPredictions (10)
    plot_pnl (c.ds)


#Buy before the close - SPX500 and DAX
if False:
    ds = Dataset(ccy_pair='SPX500_USD', 
                              from_time = 2000,
                              to_time=2019, 
                              timeframe='M15')

    ds.loadCandles ()
    ds.computeFeatures (bComputeHighLowFeatures=False, bComputeNormalizedRatios=False, bComputeIndicators=True)
    ds.computeLabels (min_stop=0.005, periods_ahead_returns=[1,2,5])
    
    c = Chromossome (ds = ds, bDebug = True, bSlim = False)
    
    c.add_gene(timeframe = 'M15',
                       func_dict = {
                                'just_before_close':{'func':is_just_before_close, 
                                                  'kwargs':{'periods_ahead':4, #multiples of 15min
                                                            'closing_time': 21
                                                            }
                                                  }                            
                                    },
                         pred_label = 'just_before_close',
                         pred_func= fn_pred_asymmetric, 
                         pred_type = 'asymmetric',   #TODO: check why this gene is defaulted to binary instead of symmetric
                         pred_kwargs = {'indic': 'just_before_close',
                                             'threshold_min': 0.5,
                                             'threshold_max': 1.5,
                                             'direction': 1 #1 for long, -1 for short
                                            })    
    
    c.add_gene(timeframe = 'W',
                     func_dict = {
                                            },
                     pred_label = 'RSI_binary',
                     pred_func= fn_pred3, 
                     pred_kwargs = {
                                    'pred_type': 'binary',
                                    'indic': 'RSI',
                                         'threshold_min': 40,
                                         'threshold_max': 75,
                                         'inv_threshold_fn': None})


    
    c.run ()
    c.ds.removeSerialPredictions (5)    
    plot_pnl (c.ds)