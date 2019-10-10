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

from fredapi import Fred

#this function loads broad reer computed by BIS from FRED (St Louis Fed)
def reer_fred_wrap (ds, **kwargs):
    for_ccy, dom_ccy = ds.ccy_pair.split ('_')    
    
    offset = parse_kwargs('offset', 2, **kwargs)
    series_dict = {'AUD': 'RBAUBIS',
                       'CNH': 'RBCNBIS',
                       'TRY': 'RBTRBIS',
                       'MXN': 'RBMXBIS',
                        'JPY': 'RBJPBIS',
                        'USD': 'RBUSBIS',
                        'CAD': 'RBCABIS',
                        'NZD': 'RBNZBIS',
                        'NOK': 'RBNOBIS',
                        'SEK': 'RBSEBIS',
                        'EUR': 'RBXMBIS',
                        'GBP': 'RBGBBIS',
                        'CHF': 'RBCHBIS',
                        'PLN': 'RBPLBIS', 
                        'HUF': 'RBHUBIS',
                        'CZK': 'RBCZBIS',
                        'ILS': '',
                        'ZAR': 'RBZABIS',
                        }
    
    for ccy in [for_ccy, dom_ccy]:
        if ccy in series_dict:
            macro_df = ds.loadMacroData (fred_series_name = series_dict [ccy], 
                                         suffix = ccy + '_reer', 
                                         offset_macro_timeframe = offset, 
                                         bReturnMacroDf = True)
            col = macro_df.columns [0]
            
            for T in [5, 10, 20]:
                macro_df['over_' + str (T)+'y_average'] = macro_df[col] / macro_df[col].shift (12).rolling(window = 12 * T).mean ()
            ds.mergeMacroDf (macro_df, suffix = ccy + '_reer')
        else:
            raise ("No REER data available for " + str (ccy))
    
    ds.f_df['reer_ratio'] = ds.f_df[series_dict [for_ccy] + '_' + for_ccy + '_reer'] / ds.f_df[series_dict [dom_ccy] + '_' + dom_ccy + '_reer']
    ds.f_df['reer_ratio_log'] = np.log (ds.f_df['reer_ratio'])
    
    for T in [5, 10, 20]:
        ds.f_df['reer_ratio_over_' + str (T) + '_average'] = ds.f_df['over_' +str (T)+ 'y_average_' + for_ccy + '_reer'] / ds.f_df['over_' +str (T)+ 'y_average_' + dom_ccy + '_reer']
        ds.f_df['reer_ratio_over_' + str (T) + '_average_log'] = np.log (ds.f_df['reer_ratio_over_' + str (T) + '_average'])
    
    delta_t = parse_kwargs (['delta_t'], 1, **kwargs)
    
    if type (delta_t) == list:
        for t in delta_t:
            ds.f_df['reer_change_' + str (t)] = np.log(ds.f_df['reer_ratio'] / ds.f_df['reer_ratio'].shift (t))
    else:
        ds.f_df['reer_change'] = np.log(ds.f_df['reer_ratio'] / ds.f_df['reer_ratio'].shift (delta_t))
        
    return #ToDo: maybe need to return smtg



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
if False:
    c = Chromossome (ds = Dataset(ccy_pair='EUR_SEK',
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

#working with FRED data
if True:
    c = Chromossome (ds = Dataset(ccy_pair='EUR_USD',
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
        
    
    
    #this gene is extremely low frequency, need to find the best use for it
    #maybe just preventer, or preventer + momentum
    if True:
        c.add_gene(timeframe = 'D',
                           func_dict = {'dummy':{'func':reer_fred_wrap,
                                                                  'kwargs':{ 
                                                                                'offset': 2                                                                           
                                                                            }
                                                                  }},
                             pred_label = 'reer_ratio_over_10_average_log',
                             pred_func= fn_pred3,
                             
                             pred_kwargs = {'indic': 'reer_ratio_over_10_average_log',
                                                 'threshold_min': -999.0,
                                                 'threshold_max': -0.15,
                                                 'inv_threshold_fn': inv_fn_symmetric})
        
    c.run ()
    c.ds.computeLabels ()
    plot_pnl (c.ds)