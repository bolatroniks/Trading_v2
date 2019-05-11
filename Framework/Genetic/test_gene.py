# -*- coding: utf-8 -*-

from Trading.Genetic.Gene import *
from Trading.Genetic.GenePCA import GenePCA
from Trading.Genetic.Chromossome import Chromossome

from Trading.Dataset.Dataset import Dataset
from Trading.Dataset.DatasetHolder import DatasetHolder

from Config.const_and_paths import *
from Trading.Strategy.Utils.strategy_func import *

import numpy as np
import gc

if False:
    genome = []
    
    for pc in ['0', '01', '012']:
        for bounds in [(-3.0, -1.0), (-2.5, -0.5), (0.5, 2.5), (1.0, 3.0)]:
            for rho_filter in [False, True]:
                c1 = Chromossome (name = 'PCA_resid_' + pc + '_only_' + str (bounds[0]) + '_' + str (bounds[1]) + '_rho_' + str (rho_filter),
                                    ds = Dataset(ccy_pair='AUD_USD', 
                                          #from_time='2015-10-01 00:00:00', 
                                          from_time = 2006,
                                          to_time=2010, 
                                          timeframe='M15'), bDebug = False, bSlim = False)
                
                c1.add_gene (GenePCA(ds = c1.get_last_timeframe_gene('M15').ds,
                                    gene_id = 'PCA_resid_' + pc,
                                    pred_label = 'PCA_resid_' + pc,
                                    pred_func = fn_pred3,
                                    pred_kwargs = {
                                                    'indic': 'n_resid_'+pc,
                                                    'threshold_min': bounds [0],
                                                    'threshold_max': bounds [1],
                                                    'inv_threshold_fn': inv_fn_symmetric
                                                    }
                                    ))
                
                if rho_filter:
                    c1.add_gene (pred_label = 'PCA_rho',
                        pred_func = fn_pred3,
                        pred_kwargs = {
                                        'indic': 'rho_' + pc,
                                                'threshold_min': 0.3,
                                                'threshold_max': 0.8,
                                                'inv_threshold_fn': None
                                        })
                
                c1.save ()
                
                genome.append (c1)
    
    for crx in genome:
        for instrument in ['AUD_USD', 'USD_ZAR', 'USD_JPY', 'SPX500_USD', 'EUR_USD', 'GBP_USD']:
            print ('Running - ' + crx.name + ' - ' + instrument + '\n')
            crx.run (instrument = instrument)
            crx.save_stats ()
            
        del crx
        gc.collect ()

#mean reversion 1
if False:
    c1 = Chromossome (ds = Dataset(ccy_pair='EUR_USD', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2013,
                              to_time=2013, 
                              timeframe='H1'), bDebug = True, bSlim = False)
    
    c1.add_gene (timeframe = 'H1', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife_H1',
                                                'threshold_min': 0, 
                                                'threshold_max': 50,
                                                'inv_threshold_fn': None
                                            })
                    
    c1.add_gene(timeframe = 'D',
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
                     pred_label = 'RSI_D',
                     pred_func= fn_pred_double, 
                     pred_kwargs = {'indic1': 'RSI_D',
                                         'threshold_min1': 30,
                                         'threshold_max1': 35,
                                         'inv_threshold_fn1': inv_fn_rsi,
                                         'indic2': 'RSI_off_low_D',
                                        'dual_indic2': 'RSI_off_high_D',
                                        'threshold_min2': 3.0,
                                        'threshold_max2': 15.0,
                                        'inv_threshold_fn2': inv_fn_symmetric #should be symmetric?
                                        })
                    
    #c.add_gene (timeframe = 'D')

    c1.run ()
    
    print (str (c1.get_stats ()))
    
#mean reversion 2
if False:
    c2 = Chromossome (ds = Dataset(ccy_pair='EUR_USD', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2010,
                              to_time=2013, 
                              timeframe='H4'), bDebug = True, bSlim = False)
    
    c2.add_gene (timeframe = 'H4', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{'window': 504}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 0, 
                                                'threshold_max': 100,
                                                'inv_threshold_fn': None
                                            })
                    
    c2.add_gene(timeframe = 'D',
                   func_dict = {
                            'close_over_50d_ma_normbyvol_off_low':{'func':feat_off_low_high, 
                                              'kwargs':{'feat': 'close_over_50d_ma_normbyvol',
                                                        'window': 20,
                                                        'high_low': 'low'}
                                              },
                            'close_over_50d_ma_normbyvol_off_high':{'func':feat_off_low_high, 
                                              'kwargs':{'feat': 'close_over_50d_ma_normbyvol',
                                                        'window': 20,
                                                        'high_low': 'high'}
                                              },
                                },
                     pred_label = 'close_over_50d_ma_normbyvol',
                     pred_func= fn_pred_double, 
                     pred_kwargs = {'indic1': 'close_over_50d_ma_normbyvol',
                                         'threshold_min1': -999.0,
                                         'threshold_max1': -0.2,
                                         'inv_threshold_fn1': inv_fn_symmetric,
                                         'indic2': 'close_over_50d_ma_normbyvol_off_low',
                                        'dual_indic2': 'close_over_50d_ma_normbyvol_off_high',
                                        'threshold_min2': 0.05,
                                        'threshold_max2': 0.15,
                                        'inv_threshold_fn2': inv_fn_symmetric #should be symmetric?
                                        })
                    
    c2.add_gene(timeframe = 'D',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 35,
                                         'threshold_max': 65,
                                         'inv_threshold_fn': None})

    c2.run ()
    
    print (str (c2.get_stats ()))
    
    plot_pnl (c2.ds)
    plot_signals (c2.ds)
    
#mean reversion 3
if False:
    c2 = Chromossome (ds = Dataset(ccy_pair='EUR_USD', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2007,
                              to_time=2010, 
                              timeframe='M15'), bDebug = True, bSlim = False)
    
    c2.add_gene (GenePCA(ds = c2.get_last_timeframe_gene('M15').ds,
                        pred_label = 'PCA_resid',
                        pred_func = fn_pred3,
                        pred_kwargs = {
                                        'indic': 'n_resid_01_M15',
                                        'threshold_min': 1.75,
                                        'threshold_max': 2.25,
                                        'inv_threshold_fn': inv_fn_symmetric
                                        }
                        ))
    
    c2.add_gene (timeframe = 'D', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{'window': 250}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 0, 
                                                'threshold_max': 50,
                                                'inv_threshold_fn': None
                                            })
    
    c2.add_gene (timeframe = 'D', func_dict = {'new_hilo':{'func':fn_new_hilo, 
                                                          'kwargs':{'window': 375}
                                                          }
                                            },
                                pred_label = 'new_hilo',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'new_hilo',
                                                'threshold_min': -0.1, 
                                                'threshold_max': 0.5,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })
    
    c2.add_gene (timeframe = 'H4', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{'window': 500}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 0, 
                                                'threshold_max': 100,
                                                'inv_threshold_fn': None
                                            })              
    
                    
    c2.add_gene(timeframe = 'D',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 35,
                                         'threshold_max': 65,
                                         'inv_threshold_fn': None})
    
    c2.add_gene(timeframe = 'H4',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 35,
                                         'threshold_max': 65,
                                         'inv_threshold_fn': None})
    

    c2.run ()
    c2.ds.removeSerialPredictions (50)
    print (str (c2.get_stats ()))
    
    plot_pnl (c2.ds)
    plot_signals (c2.ds)
    
#hilo
if False:
    c3 = Chromossome (ds = Dataset(ccy_pair='AUD_NZD', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2010,
                              to_time=2015, 
                              timeframe='M15'), 
            path = '/home/joanna/Desktop/Projects/Trading/Files/Genetic/Production', 
            bDebug = True, bSlim = False)
    
    c3.load(filename = 'Momentum_01_hilo.crx')
    
if False:
    c2 = Chromossome (ds = Dataset(ccy_pair='AUD_NZD', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2010,
                              to_time=2015, 
                              timeframe='M15'), bDebug = True, bSlim = False)
    
    
    
    c2.add_gene (timeframe = 'D', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{'window': 250}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 50, 
                                                'threshold_max': 50000,
                                                'inv_threshold_fn': None
                                            })
    
    c2.add_gene (timeframe = 'D', func_dict = {'new_hilo':{'func':fn_new_hilo, 
                                                          'kwargs':{'window': 126,
                                                                    'conv_window': 60}
                                                          }
                                            },
                                pred_label = 'new_hilo',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'new_hilo',
                                                'threshold_min': 0.5, 
                                                'threshold_max': 9999,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })
    
    c2.add_gene (timeframe = 'H4', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{'window': 500}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife',
                                                'threshold_min': 0, 
                                                'threshold_max': 200,
                                                'inv_threshold_fn': None
                                            })
        
    c2.add_gene (timeframe = 'D', 
                                pred_type = 'binary',
                                pred_label = 'close_over_ma',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'close_over_100d_ma_normbyvol',
                                                'threshold_min': -0.3, 
                                                'threshold_max': 0.3,
                                                'inv_threshold_fn': None
                                            })

# =============================================================================
#     c2.add_gene (timeframe = 'D', func_dict = {'diff_highs_lows':{'func':feats_operation, 
#                                                           'kwargs':{'feat1': 'no_standing_highs_10',
#                                                                     'feat2': 'no_standing_lows_10',
#                                                                     'operation': 'difference'}
#                                                           }
#                                             },
#                                 pred_type = 'preventer',
#                                 pred_label = 'diff_highs_lows',
#                                 pred_func = fn_over_bought_sold,
#                                 pred_kwargs = {
#                                                 'indic': 'diff_highs_lows',
#                                                 'conv_window': 1,
#                                                 'threshold_overbought': 70,
#                                                 'threshold_oversold': 30
#                                             })         
# =============================================================================
    
    c2.add_gene (timeframe = 'D', func_dict = {'diff_highs_lows':{'func':feats_operation, 
                                                          'kwargs':{'feat1': 'no_standing_highs_10',
                                                                    'feat2': 'no_standing_lows_10',
                                                                    'operation': 'difference'}
                                                          }
                                            },
                                pred_label = 'diff_highs_lows',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'diff_highs_lows',
                                                'threshold_min': -9999,
                                                'threshold_max': -5,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })
                    
    c2.add_gene(timeframe = 'D',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 30,
                                         'threshold_max': 70,
                                         'inv_threshold_fn': None})
        
    c2.add_gene(timeframe = 'W',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 30,
                                         'threshold_max': 70,
                                         'inv_threshold_fn': None})
    
    c2.add_gene(timeframe = 'H4',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 30,
                                         'threshold_max': 35,
                                         'inv_threshold_fn': inv_fn_rsi})
        
    c2.add_gene(timeframe = 'W',
                     func_dict = {'dummy':{'func':fn_over_bought_sold, 
                                          'kwargs':{'conv_window': 6,
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
    

    c2.run ()
    c2.ds.removeSerialPredictions (50)
    print (str (c2.get_stats ()))
    
    plot_pnl (c2.ds)
    plot_signals (c2.ds)

if True:
    c = Chromossome (ds = Dataset(ccy_pair='USD_ZAR', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2006,
                              to_time=2011, 
                              timeframe='M15'), bDebug = True, bSlim = True)
    
    
    c.add_gene (GenePCA(ds = c.get_last_timeframe_gene('M15').ds,
                        pca_feat_path = os.path.join (PCA_DEFAULT_PATH, 'VAR3'),
                        pred_label = 'PCA_resid',
                        pred_func = fn_pred3,
                        pred_kwargs = {
                                        'indic': 'n_resid_01_M15',
                                        'threshold_min': 1.0,
                                        'threshold_max': 2.5,
                                        'inv_threshold_fn': inv_fn_symmetric
                                        }
                        ))
    c.add_gene (GenePCA(ds = c.get_last_timeframe_gene('M15').ds,
                        pca_feat_path = PCA_DEFAULT_PATH,
                        pred_label = 'PCA_rho',
                        pred_func = fn_pred3,
                pred_kwargs = {
                                'indic': 'rho_01_M15',
                                        'threshold_min': 0.3,
                                        'threshold_max': 0.8,
                                        'inv_threshold_fn': None
                                }))
    
    c.add_gene (timeframe = 'D', func_dict = {'halflife':{'func':halflife_wrap, 
                                                          'kwargs':{}
                                                          }
                                            },
                                pred_label = 'halflife',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'halflife_D',
                                                'threshold_min': 25,
                                                'inv_threshold_fn': None
                                            })
    
    c.add_gene (timeframe = 'D', func_dict = {'diff_lines':{'func':feats_operation, 
                                                          'kwargs':{'feat1': 'no_standing_upward_lines_10',
                                                                    'feat2': 'no_standing_downward_lines_10',
                                                                    'operation': 'difference'}
                                                          }
                                            },
                                pred_label = 'diff_lines',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'diff_lines_D',
                                                'threshold_min': 1,
                                                'threshold_max': 9999,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })
    
    c.add_gene (timeframe = 'D', func_dict = {'diff_highs_lows':{'func':feats_operation, 
                                                          'kwargs':{'feat1': 'no_standing_highs_10',
                                                                    'feat2': 'no_standing_lows_10',
                                                                    'operation': 'difference'}
                                                          }
                                            },
                                pred_label = 'diff_highs_lows',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'diff_highs_lows_D',
                                                'threshold_min': -9999,
                                                'threshold_max': -5,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })
    
    c.add_gene (timeframe = 'D', func_dict = {
                                        'diff_highs_lows':{'func':feats_operation, 
                                                          'kwargs':{'feat1': 'no_standing_highs_10',
                                                                    'feat2': 'no_standing_lows_10',
                                                                    'operation': 'difference'}
                                                          },
            
                                        'diff_highs_lows_off_low':{'func':feat_off_low_high, 
                                                          'kwargs':{'feat': 'diff_highs_lows',
                                                                    'window': 65,
                                                                    'high_low': 'low'}
                                                          },
                                        'diff_highs_lows_off_high':{'func':feat_off_low_high, 
                                                          'kwargs':{'feat': 'diff_highs_lows',
                                                                    'window': 65,
                                                                    'high_low': 'high'}
                                                          },
                                            },
                                pred_label = 'diff_highs_lows_off_extreme',
                                pred_func = fn_pred_double,
                                pred_kwargs = {
                                                'indic1': 'diff_highs_lows_D',
                                                'threshold_min1': -9999,
                                                'threshold_max1': -3,
                                                'inv_threshold_fn1': inv_fn_symmetric,
                                                'indic2': 'diff_highs_lows_off_low_D',
                                                'dual_indic2': 'diff_highs_lows_off_high_D',
                                                'threshold_min2': -9999,
                                                'threshold_max2': 1.5,
                                                'inv_threshold_fn2': inv_fn_symmetric #should be symmetric?
                                            })
    
# =============================================================================
#     c.add_gene(gene_type = 'slim', feat_to_keep = ['RSI_M15', 
#                                                             'RSI_D',
#                                                             'halflife_D',                                                            'rho_01_M15',
#                                                             'n_resid_01_M15',
#                                                             'no_standing_downward_lines_10_D',
#                                                             'no_standing_upward_lines_10_D',
#                                                             'diff_lines_D',
#                                                             'close_over_200d_ma_normbyvol_D'
#                                                            ])    
# =============================================================================
    
    
    c.add_gene(timeframe = 'M15',
                   func_dict = {
                            'RSI_off_low':{'func':feat_off_low_high, 
                                              'kwargs':{'feat': 'RSI',
                                                        'window': 65,
                                                        'high_low': 'low'}
                                              },
                            'RSI_off_high':{'func':feat_off_low_high, 
                                              'kwargs':{'feat': 'RSI',
                                                        'window': 65,
                                                        'high_low': 'high'}
                                              },
                                },
                     pred_label = 'RSI',
                     pred_func= fn_pred_double, 
                     pred_kwargs = {'indic1': 'RSI',
                                         'threshold_min1': 30,
                                         'threshold_max1': 35,
                                         'inv_threshold_fn1': inv_fn_rsi,
                                         'indic2': 'RSI_off_low',
                                        'dual_indic2': 'RSI_off_high',
                                        'threshold_min2': 5.0,
                                        'threshold_max2': 15.0,
                                        'inv_threshold_fn2': inv_fn_symmetric #should be symmetric?
                                        })
    
    c.add_gene(timeframe = 'D',
                     pred_label = 'RSI_D',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI_D',
                                         'threshold_min': 30,
                                         'threshold_max': 35,
                                         'inv_threshold_fn': inv_fn_rsi})
    
    
    c.add_gene(timeframe = 'D',
                     pred_label = 'close_over_50d_ma_normbyvol',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'close_over_50d_ma_normbyvol_D',
                                         'threshold_min': 0.25,
                                         'threshold_max': 1.0,
                                         'inv_threshold_fn': inv_fn_symmetric})
    
    #c.run ()
    
    #print (str (c.to_dict ()))
    #plot_pnl (c.ds)

if False:
    c = Chromossome (ds = Dataset(ccy_pair='USD_ZAR', 
                              #from_time='2015-10-01 00:00:00', 
                              from_time = 2006,
                              to_time=2011, 
                              timeframe='M15'), bDebug = True, bSlim = True)
    
    c.add_gene(timeframe = 'D',
                     pred_label = 'RSI',
                     pred_func= fn_pred3, 
                     pred_kwargs = {'indic': 'RSI',
                                         'threshold_min': 30,
                                         'threshold_max': 35,
                                         'inv_threshold_fn': inv_fn_rsi})
    
    c.add_gene(timeframe = 'M15',
               func_dict = {
                        'RSI_off_low':{'func':feat_off_low_high, 
                                          'kwargs':{'feat': 'RSI',
                                                    'window': 65,
                                                    'high_low': 'low'}
                                          },
                        'RSI_off_high':{'func':feat_off_low_high, 
                                          'kwargs':{'feat': 'RSI',
                                                    'window': 65,
                                                    'high_low': 'high'}
                                          },
                            },
                 pred_label = 'RSI',
                 pred_func= fn_pred_double, 
                 pred_kwargs = {'indic1': 'RSI',
                                     'threshold_min1': 30,
                                     'threshold_max1': 35,
                                     'inv_threshold_fn1': inv_fn_rsi,
                                     'indic2': 'RSI_off_low',
                                    'dual_indic2': 'RSI_off_high',
                                    'threshold_min2': 5.0,
                                    'threshold_max2': 15.0,
                                    'inv_threshold_fn2': inv_fn_symmetric #should be symmetric?
                                    })
    c.run ()
    c.get_stats ()    
    print ('No predictions: ' + str (len(c.ds.p_df['pred:RSI_D'].dropna ())) + ' - ' + str (len (c.ds.l_df)))
    
    l = c.get_permutations ()
    arr = np.zeros (len (l))
    for i, crx in enumerate(l):
        crx.run ()
        arr[i] = crx.get_stats ()['Overall']['hit_ratio']