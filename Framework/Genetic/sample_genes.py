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

ds = Dataset(ccy_pair='XAU_USD', 
                              from_time = 2000,
                              to_time=2013, 
                              timeframe='M15')

#creates a chromossome
sample_genes = Chromossome (ds = ds, bDebug = True, bSlim = False)

#RSI momentum
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

#RSI oversold simple
sample_genes.add_gene(
                         gene_id = 'RSI_oversold_symmetrical',
                         timeframe = 'D',
                         pre_type = 'symmetric',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 30,
                                             'threshold_max': 35,
                                             'inv_threshold_fn': inv_fn_rsi})

#RSI oversold off low
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
#RSI overbought preventer
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


#Halflife binary indicator
sample_genes.add_gene (
                        gene_id = 'Halflife_binary',
                        timeframe = 'D', 
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

#HiLo symmetrical
sample_genes.add_gene (
                                gene_id = 'new_hilo_symmetrical',
                                timeframe = 'D', 
                                func_dict = {'new_hilo':{'func':fn_new_hilo, 
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

#HiLo preventer
sample_genes.add_gene (
                                gene_id = 'new_hilo_preventer',
                                timeframe = 'D', 
                                func_dict = {'new_hilo':{'func':fn_new_hilo, 
                                                          'kwargs':{'window': 252,
                                                                    'conv_window': 25}
                                                          }
                                            },
                                pred_label = 'new_hilo_preventer',
                                pred_type = 'preventer',
                                pred_func = fn_pred_preventer,
                                pred_kwargs = {
                                                'indic': 'new_hilo',
                                                'threshold_min': -0.5, 
                                                'threshold_max': 999.0,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })

#Candle reversal preventer
#ToDo: BUG - get_stats method ignores this gene
sample_genes.add_gene(timeframe = 'D',
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
#Candle reversal symmetric
sample_genes.add_gene(timeframe = 'D',
                     func_dict = {'dummy':{'func':fn_candle_reversal, 
                                          'kwargs':{'conv_window': 20}
                                          }
                                            },
                     pred_label = 'reversal',
                     pred_type = 'symmetric',
                     pred_func= fn_pred3, 
                     pred_kwargs = {                                    
                                    'indic': 'strong_bullish_reversals',
                                    'dual_indic': 'strong_bearish_reversals',
                                         'threshold_min': -0.5,
                                         'threshold_max': 0.5,
                                         'inv_threshold_fn': None})

#Turn of the month asymmetric
sample_genes.add_gene(timeframe = 'D',
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

#buy just before the close, asymmetric
sample_genes.add_gene(timeframe = 'M15',
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

#Candle hammer symmetric
sample_genes.add_gene(timeframe = 'D',
                     func_dict = {'hammer':{'func':fn_hammer, 
                                          'kwargs':{'conv_window': 5,
                                                    'MIN_BAR_LENGTH': 0.001,
                                                    'MIN_CANDLE_BODY_RATIO': 2.5,                                                
                                                    }
                                          }
                                            },
                     pred_label = 'hammer',
                     pred_type = 'symmetric',
                     pred_func= fn_pred3, 
                     pred_kwargs = {                                    
                                    'indic': 'hammer',                                    
                                         'threshold_min': 2.5,
                                         'threshold_max': 999.9,
                                         'inv_threshold_fn': inv_fn_symmetric})
