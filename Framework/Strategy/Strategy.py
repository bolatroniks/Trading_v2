# -*- coding: utf-8 -*-

import argparse

from Config.const_and_paths import NEUTRAL_SIGNAL, LONG_SIGNAL
import time


from Trading.Dataset.Dataset import Dataset
from Trading.Dataset.DatasetHolder import *
from Trading.Strategy.Rules import *
from Trading.Execution.Broker import *
from Trading.Execution.Broker.Order import *

#This Strategy Class does a lot of things:
#a) it can trade automatically on Oanda;
#b) it can be backtested on historical data;
#c) it can be backtested on random data => to spot lookahead bias;

class Strategy ():
    def __init__ (self, name='My_Strat', rule = None, instruments=fx_list,
                  reporting_ccy = 'USD',
                  value_per_bet = 300, max_open_positions=5, serial_gap=20):
        
        self.reporting_ccy = reporting_ccy
        self.value_per_bet = value_per_bet
        self.name = name        
        self.rule = None
        self.init_rule (rule)
        
        self.instruments = None
        self.init_instruments (instruments)
        
        self.signals = {}
        self.last_timestamp = None
        self.serial_gap = serial_gap
        self.instruments = instruments
        self.open_positions = {}
        self.max_open_positions = max_open_positions
        self.log = None
    
            
    def init_rule (self, rule = None):
        if rule is not None:
            self.rule = rule
        if self.rule is None:
            self.rule=Rule(name = 'Multiframe01_no_serial_without_lag_neutral_daily_criterium', 
                         func = rule_mtf_simpleMomentum,                          
                         args={'pred_lag':0,
                               'criterium_d_threshold_high':45,
                               'criterium_d_threshold_low':55})
    
    def init_instruments (self, instruments=None):
        if instruments is not None:
            self.instruments = instruments
            
    def reset_signals (self):
        for instrument in self.instruments:
            if instrument in self.signals.keys ():
                self.signals [instrument]['signal'] = NEUTRAL_SIGNAL

    def plot_last_signals (self, last_time_stamp=None):
        plt.figure ()
        self.open_positions = {}
        for instrument in self.instruments:
            if last_time_stamp is None:
                last_time_stamp=self.signals[instrument]['last_px_in_dataset']
            self.updateSignals(last_time_stamp = last_time_stamp, 
                               instrument=instrument)
            try:
                plt.plot(self.ds_h4.p_df.Predictions[-15:])
                if np.sum (self.ds_h4.p_df.Predictions[-15:]**2) > 0:
                    plt.plot(self.ds_h4.p_df.Predictions[-15:], label=instrument)
            except:
                pass
        plt.legend (loc='best')
        plt.show ()
        
    
    def updateSignals (self, last_time_stamp, rule=None, 
                       instrument=None, instrument_list=None):        
        
        self.reset_signals ()
        open_slots = self.get_open_slots ()
        
        if open_slots > 0:
            self.last_timestamp = last_time_stamp
            
            other_ds = []
            for tf in self.rule.other_timeframes:
                other_ds.append(Dataset(timeframe=tf, from_time=2013, to_time=self.last_timestamp))
                ds_d.initOnlineConfig ()
                
            ds = Dataset(timeframe=self.rule.timeframe, 
                            from_time='2016-01-01 00:00:00', 
                            to_time=self.last_timestamp)            
            ds.initOnlineConfig ()
            
            self.other_ds = other_ds
            self.ds = ds
            
            if instrument_list is None and instrument is None:
                instrument_list = ['EUR_USD']
            elif instrument_list is None and instrument is not None:
                instrument_list = [instrument]
            
            for ccy_pair in instrument_list:
                try:
                #if True:
                    for ds in self.other_ds:
                        ds.loadSeriesOnline (instrument=ccy_pair)
                        ds.computeFeatures (bComputeHighLowFeatures=(self.rule.bUseHighLowFeatures if ds.timeframe == 'D' else False))
                    self.ds.loadSeriesOnline (instrument=ccy_pair)
                    self.ds.computeFeatures (bComputeHighLowFeatures=False)
    
                    ds_holder = DatasetHolder(from_time=self.ds.from_time, to_time=self.ds.to_time)
                    
                    ds_holder.ds_dict = {}
                    ds_holder.ds_dict[self.ds.ccy_pair+'_'+self.ds.timeframe] = self.ds
                    for ds in self.other_ds:
                        ds_holder.ds_dict[ds.ccy_pair+'_'+ds.timeframe] = ds
                    
                    pred = self.rule.predict(ds_holder, verbose=True)
                    stop, target = self.rule.get_stop_target(self.ds.f_df)
                    self.ds.p_df = deepcopy(self.ds.f_df.ix[:, 0:6])
                    self.ds.p_df['Predictions'] = pred
                    self.ds.removeSerialPredictions (self.serial_gap)
    
                    self.signals [ccy_pair] = {'signal':pred [-1] - NEUTRAL_SIGNAL, 
                                                'stop': stop, 
                                                'target' : target,
                                                'last_px_in_dataset': self.ds.f_df.Close[-1]}
                except:
                    print ('Error updating prediction on: ' + ccy_pair)
            
    def processSignals (self):
        orders = []
        open_slots = self.get_open_slots ()
        

        for instrument in self.signals.keys ():             
            signal = self.signals [instrument] ['signal']

            if signal != NEUTRAL_SIGNAL and instrument not in self.open_positions.keys ():
                if signal == LONG_SIGNAL:
                    direction = 'buy'
                else:
                    direction = 'sell'
                
                orders.append (Order(instrument=instrument, 
                                     direction = direction,
                                     units = signal * self.get_pos_size(instrument, 
                                                                        self.signals [instrument] ['stop']),
                                        strategy = self,
                                        signal_price = self.signals[instrument]['last_px_in_dataset'],
                                        target_pct = self.signals [instrument] ['target'],
                                        stop_loss_pct = self.signals [instrument] ['stop'],
                                            ))
        
        return orders, open_slots
                                
    def get_open_slots (self):
        return self.max_open_positions - len (self.open_positions.items())
    
    #def get_last_timestamp (self):
        
        #self.last_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        #return self.last_timestamp
        
    def get_pos_size (self, instrument, stop):
        try:
            adj = {'EUR':1.2, 'GBP':1.3, 'USD':1.0, 'XAU':1300, 'AUD': 0.8, 'PLN':0.25, 'CAD':0.8}
    
            num, den = instrument.split ('_')
            if num in adj.keys ():
                factor = adj [num]
                
                
            elif den == 'USD':
                factor = self.signals[instrument]['last_px_in_dataset']
                
            else:
                if den+'_USD' in self.signals.keys ():
                    factor = self.signals[instrument]['last_px_in_dataset'] * self.signals[den+'_USD']['last_px_in_dataset']
                else:
                    factor = self.signals[instrument]['last_px_in_dataset'] / self.signals['USD_'+den]['last_px_in_dataset']
            
            size = self.value_per_bet / ( stop * factor)            
            size = np.floor(size)
        except:
            return 0
            
        return size
        
    def backtest_historical (self, instruments=None, 
                        from_time=2000, 
                        to_time=2006, 
                        rule=None,
                        bComputePred=True, bComputeHighLowFeatures=False,
                        bComputeLabels=False,
                        bRemoveSerialPredictions=True, serial_gap=10,                        
                        target_multiple=None #parameter to make stop and target asymetrical
                        ):
        self.acc_list = []
        self.neutral_list = []
        self.init_instruments (instruments)
        self.ret_list = []
        self.cum_ret = None
        
        self.init_rule (rule)
        
        if target_multiple is not None:
            self.rule.target_multiple = target_multiple
        
        if bComputePred:
            for ccy_pair in self.instruments:
                #if True:
                try:
                    self.ds_holder= DatasetHolder(from_time=from_time,
                                      to_time=to_time)
                    self.ds_holder.loadMultiFrame (ccy_pair_list=[ccy_pair])
                    self.ds_holder.alignDataframes ()            
                    
                    self.ds_d = self.ds_holder.ds_dict[ccy_pair+'_D']
                    self.ds_h4 = self.ds_holder.ds_dict[ccy_pair+'_H4']
                    self.pred = self.rule.func (self.ds_holder, args=self.rule.args, verbose=False)            
                     
                    self.ds_h4.p_df = deepcopy(self.ds_h4.f_df.ix[:, 0:6])
                    self.ds_h4.p_df['Predictions'] = self.pred
                    if bRemoveSerialPredictions:
                        print ('Removing Serial predictions')
                        self.ds_h4.removeSerialPredictions (serial_gap)
                    if bComputeLabels:
                        self.ds_h4.computeLabels (bVaryStopTarget=True,
                                                  target_fn=self.rule.target_fn,
                                                  stop_fn=self.rule.stop_fn,
                                                  target_multiple=self.rule.target_multiple)
                    #self.ds_h4.
                    #self.ds_h4.savePredictions (rule.name)
                    
                    acc = np.sum( (self.ds_h4.p_df.Predictions == self.ds_h4.l_df.Labels) * (self.ds_h4.p_df.Predictions != NEUTRAL_SIGNAL)) / np.sum(self.ds_h4.p_df.Predictions ** 2)
                    neutral =  np.float(np.sum(self.ds_h4.p_df.Predictions == NEUTRAL_SIGNAL)) / np.float(len (self.ds_h4.p_df))
                    self.acc_list.append (acc)
                    self.neutral_list.append(neutral)
                    print (str(acc) + ', ' + str (neutral))
                    
                    fig = plt.figure ()
                    a = self.ds_h4.evaluateRule (instrument=ccy_pair, rule=self.rule)
                    #for i, no in enumerate(a):
                    a[:] = np.nan_to_num(a[:])
                    plt.plot(np.cumsum(a))
                    plt.show ()
                    self.ret_list.append(np.cumsum(a))
                    
                except:
                    print ('Error processing ' + ccy_pair)
                    
    def testStrategyOnRandomSeries (self):
        self.ds_h4.randomizeCandles ()
        self.ds_h4.computeFeatures ()
        self.ds_h4.computeLabels ()
        self.ds_d.buildCandlesFromLowerTimeframe (self.ds_h4.df)
        self.ds_d.computeFeatures
        self.ds_d.computeFeatures ()
        self.ds_d.computeLabels ()
        
        pred = self.rule.predict(self.ds_holder)
        plt.plot(pred * self.ds_h4.l_df.Labels)
        plt.plot(np.cumsum(pred * self.ds_h4.l_df.Labels))
                
        default_strategies = {
                      'mtf_simple': Strategy(name='mtf_simple', 
                         rule=Rule('Multiframe01_no_serial_without_lag_neutral_daily_criterium',
                                   rule_mtf_simpleMomentum,
                                   args={'pred_lag':0,
                                    'criterium_d_threshold_high':50,
                                    'criterium_d_threshold_low':50,
                                    'dist_fast_osc_extrema': 0.0
                                    })),

                      'xo': Strategy(name='mtf_xo', 
                        rule=Rule('MTF_XO_momentum03', 
                         rule_mtf_crossoverMomentumRule, 
                         args={'trending_criterium':'RSI', #ok
                               'trending_ratio_threshold_high':30,         #ok
                                'trending_ratio_threshold_low':70,
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
                                     
                    'hilo': Strategy(name='mtf_hilo', 
                        rule=Rule('MTF_Chase_New_HiLows_35_5_15_252_20', 
                         rule_mtf_chase_new_highs_lows, 
                         args={'osc_criterium':'RSI', #ok
                               'osc_threshold_high':65,         #ok
                                'osc_threshold_low':35,
                                'osc_dist_extrema':0,
                                  'osc_lookback_window':20,   #ok
                                  'hi_lo_lookback_window' : 252,  #ok                                                                                                            
                                  'step_width' : 20
                               }))
                      }
           
if False:
    ccy_pair = 'EUR_TRY'
    ds = Dataset(ccy_pair=ccy_pair, timeframe='D', from_time=2013, to_time='2017-08-18 18:00:00')
    ds.initOnlineConfig ()
    ds.loadSeriesOnline ()
    ds.computeFeatures ()
    
    ds2 = Dataset(ccy_pair=ds.ccy_pair, timeframe='H4', from_time='2017-01-01 00:00:00', to_time='2017-08-21 20:00:00')
    ds2.initOnlineConfig ()
    ds2.loadSeriesOnline ()
    ds2.computeFeatures ()
    
    ds_holder = DatasetHolder(from_time=ds2.from_time, to_time=ds2.to_time)
    ds_holder.ds_dict = {}
    ds_holder.ds_dict[ds2.ccy_pair+'_'+ds2.timeframe] = ds2
    ds_holder.ds_dict[ds.ccy_pair+'_'+ds.timeframe] = ds
    #try:
    #    ds_holder.alignDataframes ()
    #except:
    #    pass
    
    ds_d = ds_holder.ds_dict[ccy_pair+'_D']
    ds_h4 = ds_holder.ds_dict[ccy_pair+'_H4']
    
    rule=Rule('MTF_Chase_New_HiLows_35_5_15_252_20', 
                             rule_mtf_chase_new_highs_lows,
                             ruleType='MultiTimeframe',
                             args={'osc_criterium' : 'RSI',
                                   'osc_threshold_high':65.0,
                                   'osc_threshold_low' : 35.0,
                                   'osc_dist_extrema' : 5.0,
                                   'osc_lookback_window' : 15,
                                   'hi_lo_lookback_window' : 252,
                                   'step_width' : 20}
                                   )
    #pred = rule.func(ds_holder, rule.args)
    pred = rule.predict(ds_holder, verbose=False)
    ds_h4.p_df = deepcopy(ds_h4.f_df.ix[:, 0:6])
    ds_h4.p_df['Predictions'] = pred
    ds_h4.computeLabels ()
    
    fig = plt.figure ()
    plt.plot(np.cumsum((ds_h4.p_df.Predictions - 1.0) * ds_h4.l_df.Labels))
    #plt.plot(pred)
    plt.show ()
    #plt.plot(ds_holder.ds_dict[ds.ccy_pair+'_H4'].f_df.RSI)
    #plt.plot(ds_holder.ds_dict[ds.ccy_pair+'_H4'].f_df.Close)
    #plt.plot(ds_holder.ds_dict[ds.ccy_pair+'_H4'].f_df['ma_21_close'])
    #plt.plot(ds_holder.ds_dict[ds.ccy_pair+'_H4'].f_df['close_over_21d_ma'])
    #plt.plot(ds_holder.ds_dict[ds.ccy_pair+'_H4'].f_df['hist_vol_1m_close'])