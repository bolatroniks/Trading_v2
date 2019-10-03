# -*- coding: utf-8 -*-

from Framework.Genetic.Chromossome import *
from Framework.Genetic.Gene import *


c = Chromossome (ds = Dataset(ccy_pair='AUD_USD',
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

c.save (filename = 'test01.crx', path = r'/home/joanna/Desktop/Projects/Trading/Analysis/Genetic/Files')

c2 = Chromossome(
                filename = 'test01', 
                 path = r'/home/joanna/Desktop/Projects/Trading/Analysis/Genetic/Files')

#ToDo: this line has to go
c2.load (filename = 'test01.crx', path = r'/home/joanna/Desktop/Projects/Trading/Analysis/Genetic/Files')

c.run (instrument = 'AUD_USD',
       from_time = 2008,
       to_time = 2010)

c2.run (instrument = 'AUD_USD',
       from_time = 2008,
       to_time = 2010)
a = c.ds.p_df.Predictions
b = c2.ds.p_df.Predictions
assert (np.sum(a!=NEUTRAL_SIGNAL) > 0)
assert ( np.sum(a != b) == 0)