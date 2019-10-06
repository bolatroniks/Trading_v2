# -*- coding: utf-8 -*-

from Framework.Genetic.Gene import *
from Framework.Genetic.GenePCA import GenePCA
from Framework.Genetic.Chromossome import Chromossome

from Framework.Genetic.Functions.predictors import *

from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.DatasetHolder import DatasetHolder
from Framework.Strategy.Utils.strategy_func import *
from Framework.Strategy.StrategySimulation import *

import numpy as np
import gc

    

#tests gene based on claims
if False:
    c = Chromossome (ds = Dataset(ccy_pair='SPX500_USD',                              
                                  from_time = 2000,
                                  to_time=2013,
                                  timeframe='D'), bDebug = True, bSlim = False)
    
    
    c.add_gene(timeframe = 'D',
                           func_dict = {'dummy':{'func':load_macro_data_wrap, 
                                                                  'kwargs':{'filename': 'initial_claims.csv',
                                                                            'smooth_window_fast': 4,
                                                                            'smooth_window_slow': 12,
                                                                            #'series': 'ICSA'
                                                                            }
                                                                  }},
                             pred_label = 'claims',
                             pred_func= fn_pred3, 
                             
                             pred_kwargs = {'indic': 'ICSA_change_fast_slow_macro',
                                                 'threshold_min': -999.0,
                                                 'threshold_max': -0.025,
                                                 'inv_threshold_fn': inv_fn_symmetric})
    
    c.run ()
    c.ds.computeLabels ()
    plot_pnl (c.ds)

#rule based on rates differential
if True:
    c = Chromossome (ds = Dataset(ccy_pair='EUR_JPY',
                                  from_time = 2000,
                                  to_time=2013,
                                  timeframe='D'), bDebug = True, bSlim = False)
    
    c.add_gene(timeframe = 'D',
                           func_dict = {'dummy':{'func':rates_diff_wrap,
                                                                  'kwargs':{ 
                                                                                'offset': 1                                                                           
                                                                            }
                                                                  }},
                             pred_label = 'rates_diff_change',
                             pred_func= fn_pred3,
                             
                             pred_kwargs = {'indic': 'rates_diff_change',
                                                 'threshold_min': 0.125,
                                                 'threshold_max': 999.0,
                                                 'inv_threshold_fn': inv_fn_symmetric})
    
    c.run ()
    c.ds.computeLabels ()
    plot_pnl (c.ds)
