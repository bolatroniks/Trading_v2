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

#works for gold, S&P and Nikkei
c = Chromossome (ds = Dataset(ccy_pair='XAU_USD',
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2000,
                              to_time=2019, 
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



c.add_gene(timeframe = 'H1',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 25,
                                             'threshold_max': 35,
                                             'inv_threshold_fn': inv_fn_rsi})
    

c.run ()
c.ds.computeLabels (min_stop = 0.008, target_multiple = 1.5)
#c.ds.removeSerialPredictions (10)
plot_pnl (c.ds)