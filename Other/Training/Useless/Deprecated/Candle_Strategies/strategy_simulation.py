from Trading.Dataset.DatasetHolder import *
from Framework.Miscellaneous.my_utils import *
from Trading.Dataset.DatasetHolder import *

import numpy as np


def simulateCandleStrategies (
            dsh = None,
            strategy = None,
            indic_list = [ {'indicator':'RSI_D',
                           'direction':'short_below_low',
                           'threshold_low' : 50,
                           'min_low' : 30,
                           'threshold_high' : 50,
                           'max_high' : 70,
                           },
                           {'indicator':'RSI',
                           'direction':'long_below_low',
                           'threshold_low' : 30,
                           'min_low' : 27,
                           'threshold_high' : 70,
                           'max_high' : 73,
                           },
                           {'indicator':'new_hilo_252_20_D',
                           'direction':'long_above_high',
                           'threshold_low' : -0.5,
                           'min_low' : -500,
                           'threshold_high' : 0.5,
                           'max_high' : 500,
                           }],            
            timeframe = 'H4',
            serial_gap = 10,
            bSaveResults = False,
            bComputeStatsOfWholeDataset = False,
            filename = 'Candle simulations.csv'):
    
        
        ret_list = []
        tot_preds = 0.0
        tot_hits = 0.0
        
        if strategy is not None:
            indic_list = strategy
            
        if strategy is None and indic_list is not None:
            strategy = indic_list
        
        if bComputeStatsOfWholeDataset:
            #print ('Entering whole dataset calculation')
            df = dsh.X
            labels = dsh.y
            
            pred_long = np.ones (len(labels), bool)
            pred_short = np.ones (len(labels), bool)
            
            for indic_dict in indic_list:
                if indic_dict['direction'] == 'long_below_low' or indic_dict['direction'] == 'short_above_high':
                    pred_long = pred_long & (df[indic_dict['indicator']] < indic_dict['threshold_low']) & (df[indic_dict['indicator']] > indic_dict['min_low'])
                    pred_short = pred_short & (df[indic_dict['indicator']] > indic_dict['threshold_high']) & (df[indic_dict['indicator']] < indic_dict['max_high'])
                else:
                    pred_short = pred_short & (df[indic_dict['indicator']] < indic_dict['threshold_low']) & (df[indic_dict['indicator']] > indic_dict['min_low'])
                    pred_long = pred_long & (df[indic_dict['indicator']] > indic_dict['threshold_high']) & (df[indic_dict['indicator']] < indic_dict['max_high'])
                    
            #remove serial predictions without using for loop
            pred_long = np.array(pred_long, dtype=bool)
            pred_short = np.array(pred_short, dtype=bool)
            
            for i in range(1, serial_gap):
                pred_long [i:] = pred_long[i:] & (~pred_long[0:-i])
                pred_short [i:] = pred_short[i:] & (~pred_short[0:-i])
                    
            tot_preds = np.sum(pred_long) + np.sum(pred_short)
            tot_hits = np.sum(labels[pred_long] == 1) + np.sum (labels[pred_short] == -1)
                    
        else:
            for key in dsh.ds_dict.keys ():
                if dsh.ds_dict[key].timeframe == timeframe:
                    ds = dsh.ds_dict[key]
                    df = ds.f_df
                    labels = ds.l_df.Labels
                    pred_long = np.ones (len(labels), bool)
                    pred_short = np.ones (len(labels), bool)
                    
                    for indic_dict in indic_list:
                        if indic_dict['direction'] == 'long_below_low' or indic_dict['direction'] == 'short_above_high':
                            pred_long = pred_long & (df[indic_dict['indicator']] < indic_dict['threshold_low']) & (df[indic_dict['indicator']] > indic_dict['min_low'])
                            pred_short = pred_short & (df[indic_dict['indicator']] > indic_dict['threshold_high']) & (df[indic_dict['indicator']] < indic_dict['max_high'])
                        else:
                            pred_short = pred_short & (df[indic_dict['indicator']] < indic_dict['threshold_low']) & (df[indic_dict['indicator']] > indic_dict['min_low'])
                            pred_long = pred_long & (df[indic_dict['indicator']] > indic_dict['threshold_high']) & (df[indic_dict['indicator']] < indic_dict['max_high'])                
                                    
                    ds.set_predictions(-np.array(pred_short, float) + np.array(pred_long, float))
                    ds.removeSerialPredictions(serial_gap=serial_gap)
                    ret = np.cumsum(ds.p_df.Predictions * labels)
                    ret_list.append (ret)
                    n_pred = np.sum(np.diff(ret)**2)
                    hits = ret[-1] + (n_pred - ret[-1])/2
                    tot_preds += n_pred
                    tot_hits += hits
        try:
            if tot_preds > 0:
                acc = float(tot_hits) / float(tot_preds)
                p_val = binom_test(tot_hits, tot_preds, 0.5)
                if p_val < 0.05:
                    print ('Summary: '+ str(tot_preds) + ' predictions, accuracy:' + str(acc) + ', p-val: ' + str(p_val))
            else:
                acc = np.nan
                p_val = np.nan
        except:
            raise
        
        if bSaveResults:
            try:
                if tot_preds > 0:
                    f = open (filename, 'a')
                    saved_str = str (dsh.ds_dict.keys()[0]) + '_' + str (dsh.ds_dict.keys()[-1]) + ', '  + str (dsh.from_time) + ', ' + str (dsh.to_time) + ', ' + str(tot_preds) + ', ' + str(acc) + ', ' + str(p_val)
                    
                    for i, indic_dict in enumerate(indic_list):
                        for key in indic_dict:
                            saved_str += ', ' + str(i)+'_'+str(key) + ', ' + str(indic_dict[key])
                    saved_str += '\n'
                    
                    #print (saved_str)
                    f.write (saved_str)
                    f.close ()
            except:
                raise                
        
        return tot_preds, acc, p_val
    

        
def createCdlIndicComboList ():
    indic_list_list = []

    third_indic_list = [{'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 45,
                          'min_low' : 30,
                          'threshold_high' : 55,
                          'max_high' : 70,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 30,
                          'min_low' : 0,
                          'threshold_high' : 70,
                          'max_high' : 100,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 100,
                          'min_low' : -10,
                          'threshold_high' : 0,
                          'max_high' : 110,
                          }]

    second_indic_list = [{'indicator':'RSI_D',
                          'direction':'long_below_low',
                          'threshold_low' : 45,
                          'min_low' : 30,
                          'threshold_high' : 55,
                          'max_high' : 70,
                          },
                          
                          {'indicator':'RSI_D',
                          'direction':'long_below_low',
                          'threshold_low' : 30,
                          'min_low' : 0,
                          'threshold_high' : 70,
                          'max_high' : 100,
                          },
                          
                          {'indicator':'RSI_D',
                          'direction':'long_above_high',
                          'threshold_low' : 50,
                          'min_low' : 30,
                          'threshold_high' : 50,
                          'max_high' : 70,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 45,
                          'min_low' : 30,
                          'threshold_high' : 55,
                          'max_high' : 70,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 27,
                          'min_low' : 0,
                          'threshold_high' : 73,
                          'max_high' : 100,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 30,
                          'min_low' : 0,
                          'threshold_high' : 70,
                          'max_high' : 100,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 33,
                          'min_low' : 0,
                          'threshold_high' : 67,
                          'max_high' : 100,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_below_low',
                          'threshold_low' : 35,
                          'min_low' : 0,
                          'threshold_high' : 65,
                          'max_high' : 100,
                          },
                          
                          {'indicator':'RSI',
                          'direction':'long_above_high',
                          'threshold_low' : 50,
                          'min_low' : 30,
                          'threshold_high' : 50,
                          'max_high' : 70,
                          },
                          
                          {'indicator':'21d_over_200d_ma_normbyvol_D',
                          'direction':'long_below_low',
                          'threshold_low' : -5.0,
                          'min_low' : -10.0,
                          'threshold_high' : 5.0,
                          'max_high' : 10.0,
                          },
                          
                          {'indicator':'21d_over_200d_ma_normbyvol_D',
                          'direction':'short_below_low',
                          'threshold_low' : -2.0,
                          'min_low' : -10.0,
                          'threshold_high' : 2.0,
                          'max_high' : 10.0,
                          }, 
                          
                          {'indicator':'close_over_200d_ma_normbyvol_D',
                          'direction':'long_below_low',
                          'threshold_low' : -5.0,
                          'min_low' : -10.0,
                          'threshold_high' : 5.0,
                          'max_high' : 10.0,
                          },
                          
                          {'indicator':'close_over_200d_ma_normbyvol_D',
                          'direction':'short_below_low',
                          'threshold_low' : -2.0,
                          'min_low' : -10.0,
                          'threshold_high' : 2.0,
                          'max_high' : 10.0,
                          },
                          
                          {'indicator':'close_over_100d_ma_normbyvol_D',
                          'direction':'long_below_low',
                          'threshold_low' : -5.0,
                          'min_low' : -10.0,
                          'threshold_high' : 5.0,
                          'max_high' : 10.0,
                          },
                          
                          {'indicator':'close_over_100d_ma_normbyvol_D',
                          'direction':'short_below_low',
                          'threshold_low' : -2.0,
                          'min_low' : -10.0,
                          'threshold_high' : 2.0,
                          'max_high' : 10.0,
                          }]

    for cdl in CDL_FEATURE_NAMES:
        for direction in ['long_below_low', 'long_above_high']:
               indic_list_list.append([
                          {'indicator':cdl+'_D',
                          'direction':direction,
                          'threshold_low' : -0.5,
                          'min_low' : -500,
                          'threshold_high' : 0.5,
                          'max_high' : 500,
                          }])
               
        for direction in ['long_below_low', 'long_above_high']:
            for sec_indic in second_indic_list:
                for third_indic in third_indic_list:
                    indic_list_list.append([
                          {'indicator':cdl+'_D',
                          'direction':direction,
                          'threshold_low' : -0.5,
                          'min_low' : -500,
                          'threshold_high' : 0.5,
                          'max_high' : 500,
                          },
                          sec_indic,
                          third_indic])

    return indic_list_list

class StrategySelector ():
    def __init__ (self, strategy_list=[], 
                  periods_path=default_periods_path, 
                  periods_filename = default_periods_filename):
        self.strategy_list = strategy_list
        self.acc_list = []
        self.pred_list = []
        self.p_val_list = []
        self.periods = loadPeriodsOfInterest (periods_path = periods_path,
                                              periods_filename = periods_filename)
        
    def shorten_list (self, dsh=None,  
                      period = '',
                      period_dict = {},
                      new_strategies = [],
                      pos_acc = 0.55,
                      pos_p_val = 0.05,
                      pos_num_pred = 10,
                      neg_acc = 0.45,
                      neg_p_val = 0.05,
                      neg_num_pred = 10,
                      serial_gap = 10):
        if period != '':
            if period not in self.periods.keys ():
                print ('Period not found')
                return
            
            period_dict = self.periods [period]
            
        if len(period_dict.keys ()) != 0:
            if dsh is None:
                dsh = DatasetHolder ()
            dsh.loadPeriodOfInterest (period_dict = self.periods[period])
        
        if len(dsh.ds_dict.keys ()) == 0:
            print ('Empty dataset holder')
            return
        
                
        for strategy in new_strategies:
            if strategy not in self.strategy_list:
                self.strategy_list.append (strategy)
        
        short_list = []        
        self.acc_list = []
        self.pred_list = []
        self.p_val_list = []
        
        for i, strategy in enumerate(self.strategy_list):
            preds, acc, p_val = simulateCandleStrategies(dsh, 
                                                         strategy, 
                                                         serial_gap = serial_gap, 
                                                         bComputeStatsOfWholeDataset=True, 
                                                         bSaveResults=False)
            if (preds >= pos_num_pred) and \
                (acc >= pos_acc) and \
                (p_val <= pos_p_val) and \
                not ( (preds >= neg_num_pred) and \
                     (acc <= neg_acc) and \
                     (p_val <= neg_p_val)):
                short_list.append (strategy)
                self.acc_list.append(acc)
                self.pred_list.append(preds)
                self.p_val_list.append (p_val)
        
        self.strategy_list = short_list
        return self
    
    def simulate (self, dsh=None,  
                      period = '',
                      period_dict = {},
                      serial_gap=10,
                      bComputeStatsOfWholeDataset = True
                      ):
        if period != '':
            if period not in self.periods.keys ():
                print ('Period not found')
                return
            
            period_dict = self.periods [period]
            
        if len(period_dict.keys ()) != 0:
            if dsh is None:
                dsh = DatasetHolder ()
            dsh.loadPeriodOfInterest (period_dict = period_dict)
        
        if len(dsh.ds_dict.keys ()) == 0:
            print ('Empty dataset holder')
            return
        
        self.acc_list = []
        self.pred_list = []
        self.p_val_list = []
        
        for i, strategy in enumerate(self.strategy_list):            
            t_preds, acc, p_val = simulateCandleStrategies(dsh, 
                                                           strategy, 
                                                           serial_gap = serial_gap, 
                                                           bComputeStatsOfWholeDataset=bComputeStatsOfWholeDataset, 
                                                           bSaveResults=False)
            
            self.acc_list.append(acc)
            self.pred_list.append(t_preds)
            self.p_val_list.append (p_val)
        
        return self
        
    
if False:
        for t in range (1):
            try:
                dsh = DatasetHolder(from_time='2014-07-01 00:00:00', to_time='2015-07-31 23:59:59')
                if True:
                #for n in range (0, 56, 8):
                    strategy_list = createCdlIndicComboList ()
                    dsh.loadOandaInstrumentListIntoOneDataset (instrument_list=['USD_ZAR'])            
                 
                    for strategy in strategy_list:
                        try:
                            simulateCandleStrategies (dsh, bSaveResults=True, 
                                                  indic_list=strategy, 
                                                  serial_gap=0, 
                                                  filename='Candles_mix_xv.csv', 
                                                  bComputeStatsOfWholeDataset=True)
                        except:
                            pass
            except:
                pass
        
if False:
    import pandas as pd
    
    st_df = pd.read_csv ('Candle simulations.csv')
    
    for i in range(len (st_df)):
        if float(st_df.Accuracy[i]) >= 0.55 and float(st_df.p_val[i]) <= 0.05:
            indic_list = []
            for j in range (5):
                if str(j) + '_indicator' in st_df.columns:
                    indic_list.append ({'indicator':st_df[str(j) + '_indicator'][i].replace(' ', ''),
                                       'direction':st_df[str(j) + '_direction'][i].replace(' ', ''),
                                       'threshold_low' : float(st_df[str(j) + '_threshold_low'][i]),
                                       'min_low' : float(st_df[str(j) + '_min_low'][i]),
                                       'threshold_high' : float(st_df[str(j) + '_threshold_high'][i]),
                                       'max_high' : float(st_df[str(j) + '_max_high'][i]),
                                       })
            no_pred, acc, p_val = simulateCandleStrategies(dsh_xv3, 
                                                           indic_list=indic_list, 
                                                           bSaveResults=True, 
                                                           filename = 'Candle_xv3.csv')
    
        
if False:
    dsh = DatasetHolder(from_time=2000, to_time=2010)
    dsh.loadOandaInstrumentListIntoOneDataset ()
    
    for cdl in CDL_FEATURE_NAMES:
        for direction in ['long_below_low', 'long_above_high']:
            indic_list = [ {'indicator':'RSI_D',
                           'direction':'short_below_low',
                           'threshold_low' : 50,
                           'min_low' : 30,
                           'threshold_high' : 50,
                           'max_high' : 70,
                           },
                           {'indicator':'RSI',
                           'direction':'long_below_low',
                           'threshold_low' : 30,
                           'min_low' : 27,
                           'threshold_high' : 70,
                           'max_high' : 73,
                           },
                           {'indicator':cdl,
                           'direction':direction,
                           'threshold_low' : -0.5,
                           'min_low' : -500,
                           'threshold_high' : 0.5,
                           'max_high' : 500,
                           }]
            simulateCandleStrategies (dsh, bSaveResults=True, indic_list=indic_list)
        

if False:
    for ccy_pair in instrument_list[-5:]:
        try:
            ds_holder= DatasetHolder(from_time='2003-01-01 00:00:00',
                              to_time='2017-01-01 00:00:00')
            ds_holder.loadMultiFrame (ccy_pair_list=[ccy_pair])
            ds_holder.alignDataframes ()
            
            pred = rule_mtf_simpleMomentum(ds_holder)
            ds_d = ds_holder.ds_dict[ccy_pair+'_D']
            ds_h4 = ds_holder.ds_dict[ccy_pair+'_H4']
            print (evaluate_rule(pred, ds_h4.y))
        except:
            pass

if False:
    bRandomize = False
    for ccy in fx_list[0:20]:
        try:
            dsh = DatasetHolder(from_time=2000, to_time=2010, instrument = ccy)
            dsh.loadMultiFrame ()
            ds = dsh.ds_dict[dsh.instrument + '_H4']
            
            if bRandomize:                
                ds.randomizeCandles ()
                ds.computeFeatures ()
                ds.computeLabels ()
                dsh.ds_dict[dsh.instrument + '_D'].buildCandlesFromLowerTimeframe(ds.df)
                dsh.ds_dict[dsh.instrument + '_D'].computeFeatures ()
                
            idx_rsi_minus_peak = ds.getFeatIdx(feat='RSI_close_minus_peak', 
                      func=feat_metrics, 
                      args={'lookback_window':6, 
                      'feat':'RSI', 
                      'metric':'close_minus_peak'})
            idx_rsi_minus_bottom = ds.getFeatIdx(feat='RSI_close_minus_bottom', 
                      func=feat_metrics, 
                      args={'lookback_window':6, 
                      'feat':'RSI', 
                      'metric':'close_minus_bottom'})
            
            idx_new_hilo = dsh.ds_dict[ccy + '_D'].getFeatIdx(feat='new_hilo_' + str(252) + '_' + str (20), 
                                  func=new_high_over_lookback_window, 
                                  args={'lookback_window':252, 
                                  'step_width':20,                      
                                  'feat':'Close'})
            dsh.appendTimeframesIntoOneDataset (daily_delay = 2)   
        except:
            pass
        
        
        
if False:
    from Trading.Execution.Broker import *    

    ret_list = []
    tot_preds = 0.0
    tot_hits = 0.0

    indicator1 = 'close_over_100d_ma_normbyvol_D'
    threshold_low1 = -2.5
    threshold_high1 = 2.5

    indicator2 = 'RSI'
    threshold_low2 = 70
    threshold_high2 = 30

    for cdl in CDL_FEATURE_NAMES:    
        for key in dsh.ds_dict.keys ():
            try:
                if dsh.ds_dict[key].timeframe == 'H4':
                    ds = dsh.ds_dict[key]
                    df = ds.f_df
                    labels = ds.l_df.Labels
                    long_pred = (df[cdl+'_D']<0) & (df[indicator1] <= threshold_low1) & (df[indicator2]<=threshold_low2)
    
                    short_pred = (df[cdl+'_D']>0) & (df[indicator1] >= threshold_high1) & (df[indicator2]>=threshold_high2)
                    ds.set_predictions(-np.array(short_pred, float) + np.array(long_pred, float))
                    ds.removeSerialPredictions(serial_gap=10)
                    ret = np.cumsum(ds.p_df.Predictions * labels)
                    ret_list.append (ret)
                    n_pred = np.sum(np.diff(ret)**2)
                    hits = ret[-1] + (n_pred - ret[-1])/2
                    tot_preds += n_pred
                    tot_hits += hits
            except:
                print ('Error handling ' + key)
        try:
            acc = tot_hits / tot_preds
            p_val = binom_test(tot_hits, tot_preds, 0.5)
            print ('Summary(' + cdl + '): '+ str(tot_preds) + ' predictions, accuracy:' + str(acc) + ', p-val: ' + str(p_val))
        except:
            pass
    
if False:
    from Trading.Execution.Broker import *
    
    bRandomize = True
    arr_mean = []    
    
    dsh = DatasetHolder(from_time=2000, to_time=2010)
    for ccy in fx_list[0:20]:
        if True:
            dsh.init_instrument(ccy)
            dsh.loadMultiFrame ()
            ds = dsh.ds_dict[dsh.instrument + '_H4']
            
            if bRandomize:                
                ds.randomizeCandles ()
                ds.computeFeatures ()
                ds.computeLabels ()
                dsh.ds_dict[dsh.instrument + '_D'].buildCandlesFromLowerTimeframe(ds.df)
                dsh.ds_dict[dsh.instrument + '_D'].computeFeatures ()
                
            idx_rsi_minus_peak = ds.getFeatIdx(feat='RSI_close_minus_peak', 
                      func=feat_metrics, 
                      args={'lookback_window':6, 
                      'feat':'RSI', 
                      'metric':'close_minus_peak'})
            idx_rsi_minus_bottom = ds.getFeatIdx(feat='RSI_close_minus_bottom', 
                      func=feat_metrics, 
                      args={'lookback_window':6, 
                      'feat':'RSI', 
                      'metric':'close_minus_bottom'})
            
            idx_new_hilo = dsh.ds_dict[ccy + '_D'].getFeatIdx(feat='new_hilo_' + str(252) + '_' + str (20), 
                                  func=new_high_over_lookback_window, 
                                  args={'lookback_window':252, 
                                  'step_width':20,                      
                                  'feat':'Close'})
            dsh.appendTimeframesIntoOneDataset (daily_delay = 2)
            df = ds.f_df
            labels = ds.l_df.Labels
            
            elem_mean = []

            elem_mean.append (np.mean (labels))
            elem_mean.append (np.mean (labels[df.new_hilo_252_20_D < 0]))
            elem_mean.append (np.mean (labels[df.new_hilo_252_20_D > 0]))
            elem_mean.append (np.mean (labels[(df.RSI_D < 60) & (df.RSI > 70) & (df.RSI_D > 30)]))            
            elem_mean.append (np.mean (labels[(df.RSI_D > 40) & (df.RSI < 30) & (df.RSI_D < 70)]))
            elem_mean.append (np.mean (labels[(df.RSI_D < 60) & (df.RSI > 70) & (df.RSI_close_minus_peak < -5) & (df.RSI_D > 30)]))
            elem_mean.append (np.mean (labels[(df.RSI_D > 40) & (df.RSI < 30) & (df.RSI_close_minus_bottom > 5) & (df.RSI_D < 70)]))
            elem_mean.append (np.mean (labels[(df.RSI_D < 60) & (df.RSI > 70) & (df.RSI_D > 30) & (df.new_hilo_252_20_D < 0)]))
            elem_mean.append (np.mean (labels[(df.RSI_D > 40) & (df.RSI < 30) & (df.RSI_D < 70) & (df.new_hilo_252_20_D > 0)]))
            elem_mean.append (np.mean (labels[(df.RSI_D < 60) & (df.RSI > 70) & (df.RSI_D > 30) & (df.RSI_close_minus_peak < -5) & (df.new_hilo_252_20_D < 0)]))
            elem_mean.append (np.mean (labels[(df.RSI_D > 40) & (df.RSI < 30) & (df.RSI_D < 70) & (df.RSI_close_minus_bottom > 5) & (df.new_hilo_252_20_D > 0)]))
            
            arr_mean.append (elem_mean)
        
    dsh.buildSingleDataFrameFromDict ()
    if True:
        from scipy.stats import binom_test
        
        indicator = 'close_over_100d_ma_normbyvol_D'
        threshold_low = -5.0
        threshold_high = 5.0
        
        print ('---------'+ indicator +'_' + str(threshold_low) + '_' + str (threshold_high) +'---Direction 1:-------------------')
        for cdl in CDL_FEATURE_NAMES:            
            long_hits = np.sum(dsh.y[(dsh.X[cdl+'_D']>0) & (dsh.X[indicator]<threshold_low)] == 1)
            no_longs = dsh.y[(dsh.X[cdl+'_D']>0) & (dsh.X[indicator]<threshold_low)].shape [0]
            short_hits = np.sum(dsh.y[(dsh.X[cdl+'_D']<0) & (dsh.X[indicator]>threshold_high)] == -1)
            no_shorts = dsh.y[(dsh.X[cdl+'_D']<0) & (dsh.X[indicator]>threshold_high)].shape [0]
                           
            p_val = binom_test(long_hits + short_hits, no_longs + no_shorts, 0.5)
            no_trades = no_longs + no_shorts
            if no_trades > 0:
                acc = float(long_hits + short_hits) / no_trades
            else:
                acc = 0.0
                
            if (p_val < 0.10 and np.abs(acc - 0.5) > 0.06):
                print (cdl+': '+str(no_trades) + ', ' + str(acc) + ', ' + str(p_val))
        
        print ('---------'+ indicator +'_' + str(threshold_low) + '_' + str (threshold_high) +'---Direction 2:-------------------')
        for cdl in CDL_FEATURE_NAMES:            
            long_hits = np.sum(dsh.y[(dsh.X[cdl+'_D']<0) & (dsh.X[indicator]<threshold_low)] == 1)
            no_longs = dsh.y[(dsh.X[cdl+'_D']<0) & (dsh.X[indicator]<threshold_low)].shape [0]
            short_hits = np.sum(dsh.y[(dsh.X[cdl+'_D']>0) & (dsh.X[indicator]>threshold_high)] == -1)
            no_shorts = dsh.y[(dsh.X[cdl+'_D']>0) & (dsh.X[indicator]>threshold_high)].shape [0]
                           
            p_val = binom_test(long_hits + short_hits, no_longs + no_shorts, 0.5)
            no_trades = no_longs + no_shorts
            if no_trades > 0:
                acc = float(long_hits + short_hits) / no_trades
            else:
                acc = 0.0
                
            if (p_val < 0.10 and np.abs(acc - 0.5) > 0.06):
                print (cdl+': '+str(no_trades) + ', ' + str(acc) + ', ' + str(p_val))        
        
        #except:
        #    arr_mean.append ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #    pass
        
    