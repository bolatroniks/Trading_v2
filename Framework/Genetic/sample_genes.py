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

global sample_genes

ds = Dataset(ccy_pair='USD_ZAR', 
                              from_time = 2013,
                              to_time=2013, 
                              timeframe='M15')

#creates a chromossome
sample_genes = Chromossome (ds = ds, bDebug = True, bSlim = False)

#RSI genes
sample_genes.add_gene(
                    gene_id = 'RSI_oversold_off_low_symmetrical',
                    timeframe = 'H1',
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

sample_genes.add_gene(
                     gene_id = 'RSI_overbought_preventer',
                     timeframe = 'D',
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

sample_genes.add_gene(
                         gene_id = 'RSI_momentum_symmetrical',
                         timeframe = 'D',
                         pre_type = 'symmetric',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 51,
                                             'threshold_max': 75,
                                             'inv_threshold_fn': inv_fn_rsi})

#adds a binary indicator
sample_genes.add_gene (
                        gene_id = 'Halflife_binary',
                        timeframe = 'H4', 
                        func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_type = 'binary',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 0,
                                                'threshold_max': 25,
                                                'inv_threshold_fn': None
                                            })

