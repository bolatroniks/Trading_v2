# -*- coding: utf-8 -*-

from Framework.Dataset.Dataset import *
from Config.const_and_paths import *

fast_timeframe = 'M15'
from_time = 2000
to_time = 2018

for ccy in full_instrument_list [28:]:
    try:
        ds_f = Dataset(from_time=from_time, 
                                    to_time=to_time, 
                                    ccy_pair=ccy,
                                    timeframe=fast_timeframe)
        ds_f.loadCandles ()
        
        for k, slow_timeframe in enumerate(['H1', 'H4', 'D']):
            ds_s = Dataset(from_time=from_time, 
                                    to_time=to_time, 
                                    ccy_pair=ccy,
                                    timeframe=slow_timeframe)
            ds_s.buildCandlesFromLowerTimeframe(ds_f.df, ratio=[4, 16, 96][k])
            
            ds_s.saveCandles ()
            
            ds_s.computeFeatures (bSaveFeatures=True)
            ds_s.computeLabels (bSaveLabels=True)
            
    except:
        pass

