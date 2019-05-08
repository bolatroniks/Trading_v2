#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 22:27:08 2017

@author: renato
"""

# test_math.py
from nose.tools import assert_equal
from parameterized import parameterized, param
import unittest

from Trading.Dataset.DatasetHolder import *
from Trading.Strategy.Rules import *


fx_list = ['EUR_USD', 
                       'EUR_GBP', 'GBP_USD', 
                       'USD_JPY', 'EUR_JPY',
                       'USD_CAD', 'EUR_CAD', 'GBP_CAD', 'CAD_JPY', 
                       'AUD_USD', 'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'AUD_NZD', 
                       'USD_NOK', 'EUR_NOK', 'NOK_JPY', 'GBP_NOK',
                       'USD_SEK', 'EUR_SEK', 
                       'USD_PLN', 'EUR_PLN',
                       'USD_HUF', 'EUR_HUF',
                       'USD_ZAR', 'EUR_ZAR',
                       'USD_TRY', 'EUR_TRY',
                       'USD_MXN', 'EUR_MXN',
                       'USD_THB', 'EUR_THB',
                       'USD_CNH']

class TestMultiTimeframeRule (unittest.TestCase):
    @parameterized.expand([
                           
                           
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_no_serial_without_lag', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':0})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_no_serial_with_lag', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_no_serial_with_lag_stringent_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':55,
                               'criterium_d_threshold_low':45})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_simple_momentum_no_serial_with_lag_25_looser_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':25,
                               'criterium_d_threshold_low':75})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_simple_momentum_no_serial_with_lag_30_looser_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':30,
                               'criterium_d_threshold_low':70})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_simple_momentum_no_serial_with_lag_35_looser_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':35,
                               'criterium_d_threshold_low':65})),
        
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_simple_momentum_no_serial_with_lag_much_looser_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':40,
                               'criterium_d_threshold_low':60})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_simple_momentum_no_serial_with_lag_looser_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':45,
                               'criterium_d_threshold_low':55})),
               
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01_no_serial_with_lag_bypassing_daily_criterium', 
                         rule_mtf_simpleMomentum, 
                         args={'pred_lag':1,
                               'criterium_d_threshold_high':5,
                               'criterium_d_threshold_low':95})),
                           
        #Crossover with momentum and lower timeframe reversal
        #used trending_ratio_threshold = 2.0
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_XO_momentum04_no_serial', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold_high':0.5,         #ok
                               'trending_ratio_threshold_low':2.0,         #ok
                                  'crossover_fast':'ma_21_close',   #ok
                                  'crossover_slow' : 'ma_50_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 50,    #ok
                                  'osc_threshold_slow_low' : 50,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 65,   #ok
                                  'osc_threshold_fast_low' : 35,    #ok
                                  'step_width' : 8
                               }),
                         bRemoveSerialPredictions=True, serial_gap=20),                    
                           
        #Crossover with momentum and lower timeframe reversal
        #used trending_ratio_threshold = 2.0
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_XO_momentum03', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold':2.0,         #ok
                                  'crossover_fast':'ma_50_close',   #ok
                                  'crossover_slow' : 'ma_100_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 50,    #ok
                                  'osc_threshold_slow_low' : 50,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 65,   #ok
                                  'osc_threshold_fast_low' : 35,    #ok
                                  'step_width' : 20
                               })),  
                           
        #Crossover with momentum and lower timeframe reversal
        #used trending_ratio_threshold = 2.0
        param (bComputePred = True,
               instrument_list=['EUR_USD'], 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_XO_momentum_test_convolution', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold':2.0,         #ok
                                  'crossover_fast':'ma_50_close',   #ok
                                  'crossover_slow' : 'ma_100_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 50,    #ok
                                  'osc_threshold_slow_low' : 50,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 65,   #ok
                                  'osc_threshold_fast_low' : 35,    #ok
                                  'step_width' : 60
                               })),
                           
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01', rule_mtf_simpleMomentum)),

        #regular rule config with vol ratio threshold = 1.10
        param (bComputePred = True,
               instrument_list=['USD_CAD'], 
               from_time=2000, 
               to_time=2013, 
               rule=Rule('Multiframe03', rule_mtf_complexRule)),

        #lowered vol ratio threshold to 0.90, could be even lower or dropped altogether
        param (instrument_list=fx_list, 
               from_time=2012, 
               to_time=2016, 
               rule=Rule('Multiframe04', rule_mtf_complexRule)),

        #Crossover with momentum and lower timeframe reversal
        #used trending_ratio_threshold = 2.0
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_XO_momentum01_test', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold':2.0,         #ok
                                  'crossover_fast':'ma_50_close',   #ok
                                  'crossover_slow' : 'ma_200_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 50,    #ok
                                  'osc_threshold_slow_low' : 50,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 65,   #ok
                                  'osc_threshold_fast_low' : 35,    #ok
                                  'step_width' : 60
                               })),

        #Crossover with momentum and lower timeframe reversal
        #used trending_ratio_threshold = 2.0
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_XO_momentum02', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold':0.5,         #ok
                                  'crossover_fast':'ma_50_close',   #ok
                                  'crossover_slow' : 'ma_200_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 70,    #ok
                                  'osc_threshold_slow_low' : 30,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 55,   #ok
                                  'osc_threshold_fast_low' : 45,    #ok
                                  'step_width' : 22 
                               })),

        #Crossover with momentum and lower timeframe reversal
        #used trending_ratio_threshold = 2.0
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_XO_momentum03', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold':2.0,         #ok
                                  'crossover_fast':'ma_50_close',   #ok
                                  'crossover_slow' : 'ma_100_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 50,    #ok
                                  'osc_threshold_slow_low' : 50,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 65,   #ok
                                  'osc_threshold_fast_low' : 35,    #ok
                                  'step_width' : 20
                               })),                   
        
        

        #on lower frame, buys if RSI < 45 / sells if > 55
        param (instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_Chase_New_HiLows01', rule_mtf_chase_new_highs_lows)),
        
        #on lower frame, buys if RSI < 33 and 5 above low / sells if > 67 and 5 below high
        #window width = 20
        param (instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_Chase_New_HiLows02', rule_mtf_chase_new_highs_lows)),
        
        #on lower frame, buys if RSI < 35 and 5 above low / sells if > 65 and 5 below high
        #window width 5 days only
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2006, 
               to_time=2010, 
               rule=Rule('MTF_Chase_New_HiLows_35_5_15_252_20_no_serial', 
                         rule_mtf_chase_new_highs_lows,
                         ruleType='MultiTimeframe',
                         args={'osc_criterium' : 'RSI',
                               'osc_threshold_high':65.0,
                               'osc_threshold_low' : 35.0,
                               'osc_dist_extrema' : 5.0,
                               'osc_lookback_window' : 15,
                               'hi_lo_lookback_window' : 252,
                               'step_width' : 20}
                               )),
                         
    ])
    
        
    #Perform test on historical data
    def test_2TimeframeRule (self, instrument_list=['EUR_USD'], 
                           from_time=2000, 
                           to_time=2006, 
                           rule=Rule('Multiframe01', rule_mtf_simpleMomentum),
                            bComputePred=True,                            
                            bRemoveSerialPredictions=True, serial_gap=10):
        self.acc_list = []
        self.neutral_list = []
        if bComputePred:
            for ccy_pair in instrument_list:
                #if True:
                try:
                    self.ds_holder= DatasetHolder(from_time=from_time,
                                      to_time=to_time)
                    self.ds_holder.loadMultiFrame (ccy_pair_list=[ccy_pair])
                    self.ds_holder.alignDataframes ()            
                    
                    self.ds_d = self.ds_holder.ds_dict[ccy_pair+'_D']
                    self.ds_h4 = self.ds_holder.ds_dict[ccy_pair+'_H4']
                    self.pred = rule.func (self.ds_holder, args=rule.args, verbose=False)            
                     
                    self.ds_h4.p_df = deepcopy(self.ds_h4.f_df.ix[:, 0:6])
                    self.ds_h4.p_df['Predictions'] = self.pred
                    if bRemoveSerialPredictions:
                        print ('Removing Serial predictions')
                        self.ds_h4.removeSerialPredictions (serial_gap)
                    
                    self.ds_h4.savePredictions (rule.name)
                    acc, neutral = evaluate_rule(self.pred, self.ds_h4.y)
                    self.acc_list.append (acc)
                    self.neutral_list.append(neutral)
                    print (str(acc) + ', ' + str (neutral))
                    
                    fig = plt.figure ()
                    a = self.ds_h4.evaluateRule (instrument=ccy_pair, rule=rule)
                    #for i, no in enumerate(a):
                    a[:] = np.nan_to_num(a[:])
                    plt.plot(np.cumsum(a))
                    plt.show ()
                except:
                    print ('Error processing ' + ccy_pair)
        if True:
            self.ds_holder= DatasetHolder(from_time=from_time,
                                      to_time=to_time)
            self.ds_holder.loadMultiFrame (ccy_pair_list=['EUR_USD'])
            self.ds_holder.alignDataframes ()
            self.ds_d = self.ds_holder.ds_dict['EUR_USD_D']
            self.ds_h4 = self.ds_holder.ds_dict['EUR_USD_H4']
            fig = plt.figure ()
            ret_list = []
            
            for ccy_pair in fx_list:
                try:
                    a = self.ds_h4.evaluateRule (instrument=ccy_pair, rule=rule)
                    
                    n_pred = np.sum((self.ds_h4.p_df.Predictions - 1.0)**2)
                    net = np.sum((self.ds_h4.p_df.Predictions - 1.0) * self.ds_h4.l_df.Labels)
                    
                    a[:] = np.nan_to_num(a)
                    plt.plot(np.cumsum(a))
                    ret_list.append(np.cumsum(a))
                except:
                    pass
            plt.show ()
            
            fig = plt.figure ()
            cum_ret = ret_list[0].loc[self.ds_holder.from_time:self.ds_holder.to_time]
            for ret in ret_list:
                try:
                    if str(type (ret)) == "<class 'pandas.core.series.Series'>":
                        if len (ret.loc[self.ds_holder.from_time:self.ds_holder.to_time]) > 0:
                            #dummy = ret_list[0] + ret
                            #if np.isreal(dummy[-1]):
                            cum_ret = cum_ret + ret.loc[self.ds_holder.from_time:self.ds_holder.to_time]
                except:
                    pass
                
            plt.plot(cum_ret)
            fig.savefig (t.ds_h4.predpath + '//' + t.ds_h4.feat_filename_prefix +
                         rule.name + '.png')
            
    #Perform live test
    @parameterized.expand([
                           
        param (bComputePred = True,
               bComputeHighLowFeatures = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe_Complex01', rule_mtf_complexRule)),
                           
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('Multiframe01', rule_mtf_simpleMomentum)),
        
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010, 
               rule=Rule('MTF_Chase_New_HiLows_35_5_15_252_20', 
                         rule_mtf_chase_new_highs_lows,
                         ruleType='MultiTimeframe',
                         args={'osc_criterium' : 'RSI',
                               'osc_threshold_high':65.0,
                               'osc_threshold_low' : 35.0,
                               'osc_dist_extrema' : 5.0,
                               'osc_lookback_window' : 8,
                               'hi_lo_lookback_window' : 252,
                               'step_width' : 20}
                               ),
                         serial_gap = 20),   
    
               
        
        param (bComputePred = True,
               instrument_list=fx_list, 
               from_time=2000, 
               to_time=2010,
               bComputeHighLowFeatures = True,
               rule=Rule('MTF_XO_momentum04', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'ratio_standing_up_downward_lines_10', #ok
                               'trending_ratio_threshold':2.0,         #ok
                                  'crossover_fast':'ma_21_close',   #ok
                                  'crossover_slow' : 'ma_50_close',  #ok                                                                                                            
                                  'osc_criterium_slow' : 'RSI',      #ok
                                  'osc_threshold_slow_high' : 50,    #ok
                                  'osc_threshold_slow_low' : 50,     #ok
                                  'osc_criterium_fast' : 'RSI',     #ok
                                  'osc_threshold_fast_high' : 65,   #ok
                                  'osc_threshold_fast_low' : 35,    #ok
                                  'step_width' : 8
                               })),
        ])
    def test_2TimeframeRuleOnline (self, instrument_list=['EUR_USD'], 
                           from_time=2000, 
                           to_time=2006, 
                           rule=Rule('Multiframe01', rule_mtf_simpleMomentum),
                            bComputePred=True, bComputeHighLowFeatures=False,
                            bRemoveSerialPredictions=True, serial_gap=10):
        self.acc_list = []
        self.neutral_list = []

        ds_d = Dataset(timeframe='D', from_time=2013, to_time='2017-08-18 18:00:00')
        ds_h4 = Dataset(timeframe='H4', from_time='2017-01-01 00:00:00', to_time='2017-08-21 20:00:00')
        ds_d.initOnlineConfig ()
        ds_h4.initOnlineConfig ()
        
        self.ds_d = ds_d
        self.ds_h4 = ds_h4
        
        cum_ret = None
        if bComputePred:
            for ccy_pair in instrument_list:
                try:
                #if True:
                    ds_d.loadSeriesOnline (instrument=ccy_pair)
                    ds_d.computeFeatures (bComputeHighLowFeatures=bComputeHighLowFeatures)
                    ds_h4.loadSeriesOnline (instrument=ccy_pair)
                    ds_h4.computeFeatures (bComputeHighLowFeatures=False)
    
                    ds_holder = DatasetHolder(from_time=ds_h4.from_time, to_time=ds_h4.to_time)
                    self.ds_holder = ds_holder
                    ds_holder.ds_dict = {}
                    ds_holder.ds_dict[ds_h4.ccy_pair+'_'+ds_h4.timeframe] = ds_h4
                    ds_holder.ds_dict[ds_d.ccy_pair+'_'+ds_d.timeframe] = ds_d
                    
                    pred = rule.predict(ds_holder, verbose=False)
                    ds_h4.p_df = deepcopy(ds_h4.f_df.ix[:, 0:6])
                    ds_h4.p_df['Predictions'] = pred
                    if bRemoveSerialPredictions:
                        ds_h4.removeSerialPredictions (serial_gap)
                    ds_h4.computeLabels ()
                    
                    ret = np.cumsum((ds_h4.p_df.Predictions - 1.0) * ds_h4.l_df.Labels)
                    fig = plt.figure ()
                    plt.plot(ret)
                    plt.plot(ds_h4.p_df.Predictions)
                    plt.show ()
                    
                    if cum_ret is None:
                        cum_ret = ret
                    else:
                        cum_ret += ret
                except:
                    print ('Error processing '+str(ccy_pair))
        fig = plt.figure ()
        plt.plot(cum_ret)
        plt.show ()