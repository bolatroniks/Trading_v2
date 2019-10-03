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

def load_claims_data_wrap (ds, **kwargs):
    filename = parse_kwargs ('filename', 'initial_claims.csv', **kwargs)
    feat = parse_kwargs (['feat', 'feature', 'series'], None, **kwargs)
    df = ds.loadMacroData (filename = 'initial_claims.csv', bReturnMacroDf=True,
                           offset_macro_timeframe = 1)
    
    if feat is None:
        feat = df.columns [0]
    
    #computes an additional feature in the macro dataframe
    df[feat + '_Change'] = np.log(df[feat].rolling (window = 4).mean () / df[feat].shift (4).rolling (window = 12).mean ())
    
    #merges the macro dataframe into the features dataframe
    ds.mergeMacroDf(df)
    

#tests gene based on claims
c = Chromossome (ds = Dataset(ccy_pair='SPX500_USD',                              
                              from_time = 2000,
                              to_time=2013,
                              timeframe='D'), bDebug = True, bSlim = False)


c.add_gene(timeframe = 'D',
                       func_dict = {'dummy':{'func':load_claims_data_wrap, 
                                                              'kwargs':{'filename': 'initial_claims.csv',
                                                                        'series': 'ICSA'}
                                                              }},
                         pred_label = 'claims',
                         pred_func= fn_pred3, 
                         
                         pred_kwargs = {'indic': 'ICSA_Change',
                                             'threshold_min': -999.0,
                                             'threshold_max': -0.025,
                                             'inv_threshold_fn': inv_fn_symmetric})

c.run ()
c.ds.computeLabels ()
plot_pnl (c.ds)