#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:48:44 2019

@author: Renato Barros
"""

from Framework.Dataset.Dataset import Dataset
from Config.const_and_paths import CONFIG_MACRO_DATA_PATH
from os.path import join

ds = Dataset(ccy_pair='EUR_SEK',
                                  from_time = 2000,
                                  to_time=2013,
                                  timeframe='D')


ds.loadCandles ()       #loads bars
ds.computeFeatures ()   #computes some indicators

#loads data from St Louis Federal Reserve database
ds.loadMacroData (fred_series_name = 'RBSEBIS', 
                                         suffix = 'SEK_reer', 
                                         offset_macro_timeframe = 2)

#loads data from a file stored locally
ds.loadMacroData (
        filename = 'Sweden 2-Year Bond Yield Historical Data.csv', 
                  path = join(CONFIG_MACRO_DATA_PATH, 'Rates'), 
                  suffix = 'SEK_rates', 
                  offset_macro_timeframe = 1)

ds.f_df.Price_SEK_rates.plot () #plots SEK 2y rates