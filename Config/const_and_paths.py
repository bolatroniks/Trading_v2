# -*- coding: utf-8 -*-

from os.path import join

#-----------v20 and Oanda-----------------------#
project_path = u'/home/joanna/Desktop/Projects'
project_name = 'Trading'

dataset_path = u'/home/joanna/Desktop/Projects/Trading/datasets'

V20_PATH = join(project_path, u'v20-python-master/src')
OANDA_PATH = r'/home/renato/Desktop/Projects/v20-python-master/src'
OANDA_ORDER_PATH = join(project_path, u'v20-python-master/src/order')
V20_CONF= join(project_path, project_name, u'datasets/Oanda/.v20.conf')
OANDA_ACCOUNT_ID = '101-004-4638013-001'
OANDA_MARKET_ORDER_SCRIPT = 'market.py'

FRED_API_KEY_FILENAME = join(project_path, project_name, u'datasets/Macro/.fred_api_key.txt')

C_UTILS_PATH = join(project_path, project_name, u'C_and_CPP/Csource/C_arraytest')
CPP_UTILS_PATH = join(project_path, project_name, u'C_and_CPP/cpp_utils/cpp_utils_v2/bin/Debug')
PCA_DEFAULT_PATH = join (dataset_path, u'Oanda/Fx/PCA_New')
FEATURES_DEFAULT_PATH = join (dataset_path, u'Oanda/Fx/Featured')
PARSED_MKT_DATA_DEFAULT_PATH = join (dataset_path, u'Oanda/Fx/Parsed')
LABELS_DEFAULT_PATH = join (dataset_path, u'Oanda/Fx/Labeled')
PREDS_DEFAUL_PATH = join (dataset_path, u'Oanda/Fx/Predicted')
CONFIG_PROD_RULE_PATH = join (project_path, project_name, u'Framework/Strategy/Rules/Production')
CONFIG_LOG_PATH = join(project_path, project_name, u'Analysis/Logs')

CONFIG_MACRO_DATA_PATH = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro'

#genetic algorithm
CONFIG_TEST_CHROMOSSOME_PATH = join(project_path, project_name, u'Files/Genetic')
CONFIG_TEST_GENES = join (project_path, project_name, u'/Files/Genetic/Genes/Test')
#-----------------------------------------------#


#------------------------------------------------------#
commodity_list = ['BCO_USD', 'WTICO_USD',
                         'CORN_USD',
                         'NATGAS_USD',
                         'SOYBN_USD', 'WHEAT_USD',
                         'SUGAR_USD']
                         
equity_list = ['AU200_AUD',
                         'NL25_EUR',
                         'DE30_EUR',                         
                         'FR40_EUR',
                         'UK100_GBP',
                         'JP225_USD',                         
                         'SG30_SGD',
                         'SPX500_USD',
                         'CN50_USD']

rates_list = ['UK10YB_GBP',
              'DE10YB_EUR',
              'USB02Y_USD','USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD']

fx_list = ['EUR_USD', 
                       'EUR_GBP', 'GBP_USD', 
                       'USD_JPY', 'EUR_JPY',
                       'USD_CAD', 'EUR_CAD', 'GBP_CAD', 'CAD_JPY', 
                       'AUD_USD', 'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'AUD_NZD', 
                       'USD_NOK', 'EUR_NOK',
                       'USD_SEK', 'EUR_SEK', 
                       'USD_PLN', 'EUR_PLN',
                       'USD_HUF', 'EUR_HUF',
                       'USD_ZAR', 'EUR_ZAR',
                       'USD_TRY', 'EUR_TRY',
                       'USD_MXN', 
                       'USD_THB',
                       'USD_CNH', 'USD_INR',
                       'USD_SGD']

random_list = ['USD_RANDOM' +str (i) for i in range (1,100)]

TIMEFRAME_LIST = ['D', 'H4', 'H1', 'M15']
                       
full_instrument_list = fx_list + commodity_list + equity_list + rates_list

#-----------GUIs-------------------------------------------------#

#Vectorized Strategy tuner
strats_path = project_path + u'Trading/Framework/Training/GUI/Strats'

#signals
NEUTRAL_SIGNAL = 0
LONG_SIGNAL = 1
SHORT_SIGNAL = -1

TF_LIST = ['M15', 'H1', 'H4', 'D', 'W']

#periods of interest
default_periods_path = './'
default_periods_filename = 'Periods of Interest.txt'
default_periods = {'EUR Dull Market 2013-2014': {'from_time': '2013-03-01 00:00:00',
          'instrument_list': ['EUR_USD'],
          'timeframes': ['D', 'H4'],
          'to_time': '2014-12-31 23:59:59'},

        'JPY Dull Market 2017': {'from_time': '2016-09-01 00:00:00',
          'instrument_list': ['EUR_USD'],
          'timeframes': ['D', 'H4'],
          'to_time': '2017-12-31 23:59:59'},

         'JPY_Abenomics_deval': {'from_time': '2012-10-01 00:00:00',
          'instrument_list': ['USD_JPY'],
          'timeframes': ['D', 'H4'],
          'to_time': '2013-05-01 23:59:59'},

         'Rates Bear Trap -2014-2015': {'from_time': '2013-09-01 00:00:00',
          'instrument_list': ['UK10YB_GBP', 'DE10YB_EUR', 'USB10Y_USD', 'USB30Y_USD'],
          'timeframes': ['D', 'H4'],
          'to_time': '2015-12-31 23:59:59'},

         'TRY_CBRT_deval': {'from_time': '2010-12-31 00:00:00',
          'instrument_list': ['USD_TRY'],
          'timeframes': ['D', 'H4'],
          'to_time': '2011-12-31 23:59:59'},

         'USD_ZAR_2014_2015_bull_market': {'from_time': '2014-07-01 00:00:00',
          'instrument_list': ['USD_ZAR'],
          'timeframes': ['D', 'H4'],
          'to_time': '2015-07-31 23:59:59'},

         'USD_bull_market': {'from_time': '2014-07-01 00:00:00',
          'instrument_list': ['USD_ZAR', 'USD_CAD', 'USD_NOK', 'USD_TRY', 'USD_JPY'],
          'timeframes': ['D', 'H4'],
          'to_time': '2015-07-31 23:59:59'}}