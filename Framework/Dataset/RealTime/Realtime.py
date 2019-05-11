# -*- coding: utf-8 -*-


#from Trading.Training.TradingModel import *
from Trading.FeatureExtractors.Bespoke.Trend.TrendlineFeaturesExtractor import *
from Trading.FeatureExtractors.Technical.indicators import get_TA_CdL_Func_List
from Framework.Miscellaneous.my_utils import printProgressBar

#from Trading.Oanda.candlesv2 import *
#from Trading.RealTime.Realtime import *
#from Trading.Training.TradingModel import TradingModel
from Trading.Dataset.dataset_func import *
from Config.const_and_paths import *

func_list = get_TA_CdL_Func_List ()

import sys
import time
import talib as ta
import numpy as np
from copy import deepcopy
import re

#sys.path.append ('/home/Downloads/v20-python-samples-master/src')
sys.path.append (V20_PATH)
sys.path.append(C_UTILS_PATH)
from _C_arraytest import *

def get_labels_suffix (bVaryStopTarget, min_stop, vol_denominator, target_multiple, stop_fn, target_fn):    
    #suffix: to be appended to the df columns so that each different set of parameters result in different columns 
    suffix = '_' + str (bVaryStopTarget) + \
                '_' + str (min_stop) + \
                '_' + str (vol_denominator) + \
                '_' + str (target_multiple)
    
    if stop_fn is not None:
        suffix += '_' + str(stop_fn.__name__)
    else:
        suffix += '_None'
    
    if target_fn is not None:
        suffix += '_' + str(target_fn.__name__)
    else:
        suffix += '_None'
        
    return suffix

def computeLabelsOnTheFly_old (my_df, target = 0.025, stop = 0.025, 
                           bSave=False, filename='',
                           bVaryStopTarget=True, min_stop = 0.01,
                           vol_denominator = 5.0, 
                           stop_fn = None, target_fn = None, 
                           target_multiple = 1.0, bOpenClosePxForEntry=False,
                           verbose=True):  
    labels = np.zeros (len(my_df['Close']))
    entries = np.zeros (len(my_df['Close']))
    profit_buy_exits = np.zeros (len(my_df['Close']))
    loss_buy_exits = np.zeros (len(my_df['Close']))
    profit_sell_exits = np.zeros (len(my_df['Close']))
    loss_sell_exits = np.zeros (len(my_df['Close']))
    
    stops = np.zeros (len(my_df['Close']))
    targets = np.zeros (len(my_df['Close']))    
    time_till_stops = np.zeros (len(my_df['Close']))
    time_till_targets = np.zeros (len(my_df['Close']))
    
    
    for i in range (len(my_df['Close']) - 1):
        if np.mod(i, 1000) ==0 and verbose:
            print ('Computing labels '+str(i)+'/'+str(len(my_df['Close'])))
        if bOpenClosePxForEntry:
            entry_px = my_df['Close'][i] #use close px
        else:
            entry_px = my_df['Open'][i+1] #use open px of the next candle for entry
            
        entries [i] = entry_px
        
        if bVaryStopTarget:
            stop = np.maximum(my_df['hist_vol_1m_close'][i] / vol_denominator, min_stop)
            target = stop * target_multiple
            if stop_fn is not None:
                stop = stop_fn (my_df, bLive=True, vol_denominator=vol_denominator, 
                                min_stop=min_stop, idx=i)
            if target_fn is not None:
                target = target_fn (my_df, bLive=True, vol_denominator=vol_denominator, 
                                min_target=min_stop * target_multiple, idx=i, 
                                target_multiple=target_multiple)
            
            stops [i] = stop
            targets [i] = target
        
    #tests if it is a good idea to buy
        for j in range (i+1, len(my_df['Close']), 1):
            bad_ret = my_df['Low'][j] / entry_px - 1
            good_ret = my_df['High'][j] / entry_px - 1
            #tests stops
            if bad_ret <= -stop:
                time_till_stops [i] = j - i
                loss_buy_exits [i] = my_df['Low'][j]
                #labels[i] = 0
                break
            if good_ret >= target:
                time_till_targets [i] = j - i
                profit_buy_exits [i] = my_df['High'][j]
                labels[i] = 1
                break
    
    #tests if it is a good idea to sell
        for j in range (i+1, len(my_df['Close']), 1):
            bad_ret = my_df['High'][j] / entry_px - 1
            good_ret = my_df['Low'][j] / entry_px - 1
            #tests stops
            if bad_ret >= stop:
                #print ('Stopped at: ', my_df['High'][j])
                #labels[i] = 0
                loss_sell_exits [i] = my_df['High'][j]
                time_till_stops [i] = j - i
                break
            if good_ret <= -target:
                #print ('Hit target at: ', my_df['Low'][j])
                time_till_targets [i] = j - i
                profit_sell_exits [i] = my_df['Low'][j]
                labels[i] = -1
                break
    
    #suffix: to be appended to the df columns so that each different set of parameters result in different columns
    suffix = get_labels_suffix (bVaryStopTarget, min_stop, vol_denominator, target_multiple, stop_fn, target_fn)
        
    ret_dict = {}
    ret_dict ['Labels' + suffix] = labels
    ret_dict ['Entries' + suffix] = entries
    ret_dict ['Profit_Buy_Exits' + suffix] = profit_buy_exits
    ret_dict ['Loss_Buy_Exits' + suffix] = loss_buy_exits
    ret_dict ['Profit_Sell_Exits' + suffix] = profit_sell_exits
    ret_dict ['Loss_Sell_Exits' + suffix] = loss_sell_exits
    ret_dict ['Stops' + suffix] = stops
    ret_dict ['Targets' + suffix] = targets
    ret_dict ['Time_till_stop' + suffix] = time_till_stops
    ret_dict ['Time_till_target' + suffix] = time_till_targets    
    
    my_df['Labels'] = labels
    
    if verbose:
        print("Longs: " + str(np.count_nonzero(labels[:]==1)))
        print("Shorts: " + str(np.count_nonzero(labels[:]==-1)))
        print("Neutral: " + str(np.count_nonzero(labels[:]==0)))
    #-------------------------------------------------------------------------#
    if bSave:
        print ('To be implemented')
        try:
            my_df.to_csv(label_path+"/"+label_filename)
            plt.figure ()
            my_df['Close'][-show_last:].plot(label="Close", legend=True)
            my_df['Labels'][-show_last:].plot(secondary_y=True, label="Label", legend=True)
            plt.legend()
            plt.show ()
        except:
            pass
    return ret_dict

def computeLabelsOnTheFly (my_df, tf_suffix = '',
                           target = 0.025, stop = 0.025, 
                           bSave=False, filename='',
                           bVaryStopTarget=True, min_stop = 0.01,
                           vol_denominator = 5.0, 
                           stop_fn = None, target_fn = None, 
                           target_multiple = 1.0, bOpenClosePxForEntry=False,
                           verbose=True):
    
    mat = np.zeros ((15, len(my_df['Close' + tf_suffix])))
    mat[0,:] = my_df['Open' + tf_suffix]
    mat[1,:] = my_df['High' + tf_suffix]
    mat[2,:] = my_df['Low' + tf_suffix]
    mat[3,:] = my_df['Close' + tf_suffix]
    mat[4,:] = my_df['hist_vol_1m_close' + tf_suffix]
    
    compute_labels (mat, min_stop, vol_denominator, target_multiple, False, True)
    
    [labels, #5
     entries, #6
     profit_buy_exits, #7
     loss_buy_exits, #8
     profit_sell_exits, #9
     loss_sell_exits, #10
     stops, #11
     targets, #12
     time_till_stops, #13
     time_till_targets #14
     ] = [mat[j,:] for j in range(5, mat.shape[0])]
        
    #suffix: to be appended to the df columns so that each different set of parameters result in different columns
    suffix = get_labels_suffix (bVaryStopTarget, min_stop, vol_denominator, target_multiple, stop_fn, target_fn)
        
    ret_dict = {}
    ret_dict ['Labels' + suffix] = labels
    ret_dict ['Entries' + suffix] = entries
    ret_dict ['Profit_Buy_Exits' + suffix] = profit_buy_exits
    ret_dict ['Loss_Buy_Exits' + suffix] = loss_buy_exits
    ret_dict ['Profit_Sell_Exits' + suffix] = profit_sell_exits
    ret_dict ['Loss_Sell_Exits' + suffix] = loss_sell_exits
    ret_dict ['Stops' + suffix] = stops
    ret_dict ['Targets' + suffix] = targets
    ret_dict ['Time_till_stop' + suffix] = time_till_stops
    ret_dict ['Time_till_target' + suffix] = time_till_targets    
    
    my_df['Labels'] = labels
    
    if verbose:
        print("Longs: " + str(np.count_nonzero(labels[:]==1)))
        print("Shorts: " + str(np.count_nonzero(labels[:]==-1)))
        print("Neutral: " + str(np.count_nonzero(labels[:]==0)))
    #-------------------------------------------------------------------------#
    if bSave:
        print ('To be implemented')
        try:
            my_df.to_csv(label_path+"/"+label_filename)
            plt.figure ()
            my_df['Close'][-show_last:].plot(label="Close", legend=True)
            my_df['Labels'][-show_last:].plot(secondary_y=True, label="Label", legend=True)
            plt.legend()
            plt.show ()
        except:
            pass
    return ret_dict

def computeFeaturesOnTheFly (df, 
                             rolling_window=60, 
                             lookback_window=252, 
                             timeframe='D',
                             bComputeIndicators=True,
                             bComputeNormalizedRatios=True,
                             bComputeCandles=True,
                             bComputeHighLowFeatures=None,
                             high_low_feat_window = 500,
                             verbose=False):

    if bComputeIndicators is None:
         if timeframe == 'W' or timeframe == 'D' or timeframe == 'H4' or timeframe == 'H1' or timeframe == 'M15':
             bComputeIndicators = True
         else:
             bComputeIndicators = False
             
    if bComputeNormalizedRatios is None:
         if timeframe == 'W' or timeframe == 'D' or timeframe == 'H4':
             bComputeNormalizedRatios = True
         else:
             bComputeNormalizedRatios = False
             
    if bComputeCandles is None:
         if timeframe == 'W' or timeframe == 'D' or timeframe == 'H4':
             bComputeCandles = True
         else:
             bComputeCandles = False

    if bComputeHighLowFeatures is None:
        if timeframe == 'D' or timeframe == 'W':
            bComputeHighLowFeatures = True
        else:
            bComputeHighLowFeatures = False        
            
    my_df = deepcopy (df)
    if True:
        print ('Starting Features computation')
        t = time.time ()
    #-----moving averages----------------#
        my_df['ma_21_close'] = my_df['Close'].rolling(window=21).mean()
        my_df['ma_50_close'] = my_df['Close'].rolling(window=50).mean()
        my_df['ma_100_close'] = my_df['Close'].rolling(window=100).mean()
        my_df['ma_200_close'] = my_df['Close'].rolling(window=200).mean()
        
        #_____hist vol_______
        
        if timeframe == 'D':
            ann_factor = (252.0/1.0) ** 0.5
            window_factor = 1.0
        elif timeframe == 'W':
            ann_factor = (252.0/7.0) ** 0.5
            window_factor = 0.2
        elif timeframe == 'M':
            ann_factor = (252.0/21.0) ** 0.5
            window_factor = 1.0/22.0
        elif timeframe == 'H4':
            ann_factor = (252.0/0.25) ** 0.5
            window_factor = 6.0
        elif timeframe == 'H1':
            ann_factor = (252.0/0.0625) ** 0.5
            window_factor = 24.0
        elif timeframe == 'M15':
            ann_factor = (252.0*4/0.0625) ** 0.5
            window_factor = 24.0 * 4
        
        my_df['hist_vol_2wk_close'] = my_df['Change'].rolling(window=int(10 * window_factor)).std() * ann_factor 
        my_df['hist_vol_1m_close'] = my_df['Change'].rolling(window=int(22 * window_factor)).std() * ann_factor
        my_df['hist_vol_3m_close'] = my_df['Change'].rolling(window=int(66 * window_factor)).std() * ann_factor
        my_df['hist_vol_6m_close'] = my_df['Change'].rolling(window=int(126 * window_factor)).std() * ann_factor
        my_df['hist_vol_1y_close'] = my_df['Change'].rolling(window=int(252 * window_factor)).std() * ann_factor
        #---------------------------------------------------------------------------#
        
        #-------------some ta-lib functions------------------------------------#        
        close = np.asarray(my_df.Close)
        high = np.asarray(my_df.High)
        low = np.asarray(my_df.Low)
        open_a = np.asarray(my_df.Open)
        
        ma_21 = np.asarray(my_df.ma_21_close)
        ma_50 = np.asarray(my_df.ma_50_close)
        ma_100 = np.asarray(my_df.ma_100_close)
        ma_200 = np.asarray(my_df.ma_200_close)
        
        #------------ratios of close vs moving averages-----------------------#
        my_df['close_over_21d_ma'] = close / ma_21
        my_df['close_over_50d_ma'] = close / ma_50
        my_df['close_over_100d_ma'] = close / ma_100
        my_df['close_over_200d_ma'] = close / ma_200

        #-----------ratios of moving averages-------------------------------#
        my_df['21d_over_100d_ma'] = ma_21 / ma_100
        my_df['21d_over_200d_ma'] = ma_21 / ma_200
        my_df['50d_over_200d_ma'] = ma_50 / ma_200

        hist_vol_2w = np.asarray(my_df.hist_vol_2wk_close)
        hist_vol_1m = np.asarray(my_df.hist_vol_1m_close)
        hist_vol_3m = np.asarray(my_df.hist_vol_3m_close)
        hist_vol_6m = np.asarray(my_df.hist_vol_6m_close)
        hist_vol_1y = np.asarray(my_df.hist_vol_1y_close)
        #if 'Volume' in my_df.columns == False:
        #    my_df['Volume'] = np.zeros (len(close))
        
        #----------ratios of volatilities-----------------------#
        my_df['2w_over_3m_vol_ratio'] = hist_vol_2w / hist_vol_3m
        my_df['2w_over_6m_vol_ratio'] = hist_vol_2w / hist_vol_6m
        my_df['1m_over_1y_vol_ratio'] = hist_vol_1m / hist_vol_1y
        
        print ('Computed simple ratios and vols in ' + str (time.time() - t))
        t = time.time ()
        
        if bComputeIndicators:
            my_df['AD'] = np.zeros(len(close)) # requires volume
            my_df['ADOSC'] = np.zeros(len(close)) # requires volume
            my_df['ADX'] = ta.ADX(high, low, close)
            my_df['ADXR'] = ta.ADX(high, low, close)
            my_df['APO'] = ta.APO(close)
    
            a = ta.AROON(high, low)
            my_df['AROON_a'] = a[0]
            my_df['AROON_b'] = a[1]
            my_df['AROONOSC'] = ta.AROONOSC(high, low)
    
            my_df['ATR'] = ta.ATR(high, low, close)
    
            a = ta.BBANDS(close)
            my_df['BBANDS_a'] = a[0]
            my_df['BBANDS_b'] = a[1]
            my_df['BBANDS_c'] = a[2]
            
            my_df['BOP'] = ta.BOP(open_a, high, low, close)
            my_df['CCI'] = ta.CCI(high, low, close)
            my_df['CMO'] = ta.CMO(close)
            my_df['DX'] = ta.DX(high, low, close)
            my_df['HT_DCPERIOD'] = ta.HT_DCPERIOD(close)
            my_df['HT_DCPHASE'] = ta.HT_DCPHASE(close)
            a = ta.HT_PHASOR(close)
            my_df['HT_PHASOR_a'] = a[0]
            my_df['HT_PHASOR_a'] = a[1]
            a = ta.HT_SINE(close)
            my_df['HT_SINE'] = a[0]
            my_df['HT_SINE'] = a[1]
            a = ta.HT_TRENDLINE(close)
            my_df['HT_TRENDLINE'] = a
            my_df['HT_TRENDMODE'] = ta.HT_TRENDMODE(close)
            my_df['KAMA'] = ta.KAMA(close)
            
            #----------MACD--------------------
            a = ta.MACD(np.array(close))
            my_df['MACD_a'] = a[0]
            my_df['MACD_b'] = a[1]
            my_df['MACD_c'] = a[2]
            a = ta.MACDEXT(close)
            my_df['MACDEXT_a'] = a[0]
            my_df['MACDEXT_b'] = a[1]
            my_df['MACDEXT_c'] = a[2]
            a = ta.MACDFIX(close)
            my_df['MACDFIX_a'] = a[0]
            my_df['MACDFIX_b'] = a[1]
            my_df['MACDFIX_c'] = a[2]
            #------------------------------------
            a = ta.MAMA(close)
            my_df['MAMA_a'] = a[0]
            my_df['MAMA_b'] = a[1]
            my_df['MAX_CLOSE'] = ta.MAX(close, timeperiod=lookback_window)
            my_df['MAX_HIGH'] = ta.MAX(high, timeperiod=lookback_window)
            my_df['MIN_CLOSE'] = ta.MIN(close, timeperiod=lookback_window)
            my_df['MIN_LOW'] = ta.MIN(low, timeperiod=lookback_window)
            my_df['MAX_INDEX_close'] = ta.MAXINDEX(close,lookback_window) / lookback_window
            my_df['MAX_INDEX_high'] = ta.MAXINDEX(high,lookback_window) / lookback_window
            my_df['MIN_INDEX_close'] = ta.MININDEX(close,lookback_window) / lookback_window
            my_df['MIN_INDEX_high'] = ta.MININDEX(low,lookback_window) / lookback_window
            my_df['MEDIAN_PRICE'] = ta.MEDPRICE(high, low)
            my_df ['MFI'] = np.zeros(len(close)) # requires volume
            my_df['MIDPOINT'] = ta.MIDPOINT(close, lookback_window)
            my_df['MIDPRICE'] = ta.MIDPRICE(high, low, lookback_window)
            my_df['MINUS_DI'] = ta.MINUS_DI(high, low, close, lookback_window)
            my_df['MINUS_DM'] = ta.MINUS_DM(high, low, lookback_window)
            my_df['MOM'] = ta.MOM(close, lookback_window)
            my_df['NATR'] = ta.NATR(high, low, close, lookback_window)
            my_df['OPV'] = np.zeros(len(close)) # requires volume
            my_df['PLUS_DI'] = ta.PLUS_DI(high, low, close, lookback_window)
            my_df['PLUS_DM']= ta.PLUS_DM(high, low, lookback_window)
            my_df['PPO'] = ta.PPO(close)
            my_df['ROC'] = ta.ROC(close, lookback_window)
            my_df['ROCP'] = ta.ROCP(close, lookback_window)
            my_df['ROCR'] = ta.ROCR(close, lookback_window)
            my_df['RSI'] = ta.RSI(close)
            my_df['SAR'] = ta.SAR(high, low)
            my_df['SAREXT'] = ta.SAREXT(high, low)
            a = ta.STOCH(high, low, close)
            my_df['STOCH_a'] = a[0]
            my_df['STOCH_a'] = a[1]
            a = ta.STOCHF(high, low, close)
            my_df['STOCHF_a'] = a[0]
            my_df['STOCHF_a'] = a[1]
            a = ta.STOCHRSI(close, lookback_window)
            my_df['STOCHRSI_a'] = a[0]
            my_df['STOCHRSI_a'] = a[1]
            my_df['TRANGE'] = ta.TRANGE(high, low, close)
            my_df['ULTOSC'] = ta.ULTOSC(high, low, close, lookback_window)
            my_df['WCLPRICE'] = ta.WCLPRICE(high, low, close)
            my_df['WILLR'] = ta.WILLR(high, low, close, lookback_window)
            
            print ('Computed indicators in ' + str (time.time() - t))
            t = time.time ()
        
        #-------------Candle Stick Patterns--------------------#
        if bComputeCandles:
            func_list = get_TA_CdL_Func_List ()
            
            for func in func_list:
                a = str (func)                
                func_name = a[a.find('CDL'):].split(' ')[0]
                func_name = re.sub('[^A-Za-z0-9]+', '', func_name)
                my_df[func_name] = func (open_a, high, low,
                                            close) / 100
            print ('Computed candle patterns in ' + str (time.time() - t))
            t = time.time ()
        #-----------------------------------------------------#
        
        #compute new features based on highs-lows
        if bComputeHighLowFeatures:
            x = np.array(my_df['Close'])
            x_rsi = np.array (my_df['RSI'])
            for relevant_threshold in [10]:
                score_list_lhll = []
                score_list_hhhl = []            
                rsi_score_list_hhhl = []
                rsi_score_list_lhll = []
                dist_standing_high_list = []
                dist_standing_low_list = []
                dist_relevant_low_list = []
                dist_relevant_high_list = []
                no_standing_highs_list = []
                no_standing_lows_list = []
                no_standing_upward_lines_list = []
                no_standing_downward_lines_list = []
            
                seg_length = high_low_feat_window
                
                #print ("Series no: "+str(series_no)+' thres: '+str(relevant_threshold))
                
                for i in range (0, len(x)):
                    if np.mod(i, 100) ==0 and True: #verbose:
                        printProgressBar (i+1, len(x), 
                                          suffix = 'Computing High Low(' + str (relevant_threshold) + '): ',
                                          length = 50)
                        #print ('Processing '+str(i))
                    if i >= seg_length:
                        tfe = TrendlineFeaturesExtractor (x=x[i-seg_length:i])
                        tfe.ts.identifyRecentHighLows(seg_length,relevant_threshold,False)
                        #score_list.append(tfe.getHHHLScore())
                        a, b = tfe.getHHHLScoreV2()
                        score_list_hhhl.append(a)
                        score_list_lhll.append(b)
                        
                        tfe2 = TrendlineFeaturesExtractor (x=x_rsi[i-seg_length:i])
                        tfe2.ts.identifyRecentHighLows(seg_length,relevant_threshold,False)        
                        a, b = tfe2.getHHHLScoreV2()
                        rsi_score_list_hhhl.append(a)
                        rsi_score_list_lhll.append(b)
                        
                        tfe.ts.checkStandingHighsLows ()
                        no_standing_highs_list.append (len(tfe.ts.standing_highs_list))
                        no_standing_lows_list.append (len(tfe.ts.standing_lows_list))
                        
                        tfe.ts.checkStandingLines(True)
                        no_standing_upward_lines_list.append (tfe.ts.upward_lines_status_list.count(True))
                        tfe.ts.checkStandingLines(False)
                        no_standing_downward_lines_list.append (tfe.ts.downward_lines_status_list.count(True))        
                        
                        a, b, c, d = tfe.getCloserRelevantHighLow (exclude_last_pts=50)
                        dist_relevant_high_list.append (c)
                        dist_relevant_low_list.append (d)
                        
                        tfe.ts.checkStandingHighsLows ()
                        a, b, c, d = tfe.getCloserStandingHighLow()
                        dist_standing_high_list.append (c)
                        dist_standing_low_list.append (d)
                    
                    else:
                        score_list_lhll.append (0)
                        score_list_hhhl.append (0)
                        rsi_score_list_hhhl.append (0)
                        rsi_score_list_lhll.append (0)
                        dist_standing_high_list.append (0)
                        dist_standing_low_list.append (0)
                        dist_relevant_low_list.append (0)
                        dist_relevant_high_list.append (0)
                        no_standing_highs_list.append (0)
                        no_standing_lows_list.append (0)
                        no_standing_upward_lines_list.append (0)
                        no_standing_downward_lines_list.append (0)
                my_df['score_list_lhll_'+str(relevant_threshold)] = score_list_lhll
                my_df['score_list_hhhl_'+str(relevant_threshold)] = score_list_hhhl
                my_df['rsi_score_list_hhhl_'+str(relevant_threshold)] = rsi_score_list_hhhl
                my_df['rsi_score_list_lhll_'+str(relevant_threshold)] = rsi_score_list_lhll
                my_df['dist_standing_high_'+str(relevant_threshold)] = dist_standing_high_list
                my_df['dist_standing_low_'+str(relevant_threshold)] = dist_standing_low_list
                my_df['dist_relevant_low_'+str(relevant_threshold)] = dist_relevant_low_list
                my_df['dist_relevant_high_'+str(relevant_threshold)] = dist_relevant_high_list
                my_df['no_standing_highs_'+str(relevant_threshold)] = no_standing_highs_list
                my_df['no_standing_lows_'+str(relevant_threshold)] = no_standing_lows_list
                my_df['no_standing_upward_lines_'+str(relevant_threshold)] = no_standing_upward_lines_list
                my_df['no_standing_downward_lines_'+str(relevant_threshold)] = no_standing_downward_lines_list
                      
                my_df['ratio_standing_up_downward_lines_'+str(relevant_threshold)] = np.asarray(no_standing_upward_lines_list,dtype=float) / np.asarray(no_standing_downward_lines_list,dtype=float)
                my_df['ratio_standing_highs_lows_'+str(relevant_threshold)] = np.asarray(no_standing_highs_list,dtype=float) / np.asarray(no_standing_lows_list,dtype=float)
                my_df['ratio_dist_relevant_highs_lows_'+str(relevant_threshold)] = np.asarray(dist_relevant_high_list,dtype=float) / np.asarray(dist_relevant_low_list,dtype=float)
                my_df['ratio_dist_standing_highs_lows_'+str(relevant_threshold)] = np.asarray(dist_standing_high_list,dtype=float) / np.asarray(dist_standing_low_list,dtype=float)
                
                
    
                my_df['dist_standing_high_normbyvol'+str(relevant_threshold)] = dist_standing_high_list / hist_vol_3m
                my_df['dist_standing_low_normbyvol'+str(relevant_threshold)] = dist_standing_low_list / hist_vol_3m
                my_df['dist_relevant_low_normbyvol'+str(relevant_threshold)] = dist_relevant_low_list / hist_vol_3m
                my_df['dist_relevant_high_normbyvol'+str(relevant_threshold)] = dist_relevant_high_list / hist_vol_3m
                
            print ('Computed high_low features in ' + str (time.time() - t))
            t = time.time ()
            
        #------------ratios of close vs moving averages normalized by vol-----------------------#                  
        if bComputeNormalizedRatios:
            
            my_df['close_over_21d_ma_normbyvol'] = (close - ma_21) / (hist_vol_3m * ma_200)
            my_df['close_over_50d_ma_normbyvol'] = (close - ma_50) / (hist_vol_3m * ma_200)
            my_df['close_over_100d_ma_normbyvol'] = (close - ma_100) / (hist_vol_3m * ma_200)
            my_df['close_over_200d_ma_normbyvol'] = (close - ma_200) / (hist_vol_3m * ma_200)
    
            #-----------ratios of moving averages normalized by vol-------------------------------#
            my_df['21d_over_100d_ma_normbyvol'] = (ma_21 - ma_100) / (hist_vol_3m * ma_200)
            my_df['21d_over_200d_ma_normbyvol'] = (ma_21 - ma_200) / (hist_vol_3m * ma_200)
            my_df['50d_over_200d_ma_normbyvol'] = (ma_50 - ma_200) / (hist_vol_3m * ma_200)
            
            print ('Computed normalized ratios in ' + str (time.time() - t))
            t = time.time ()

        #normalize the candle prices
        #my_df['Open'] = my_df['Open']
        #my_df['High'] = my_df['High']
        #my_df['Low'] = my_df['Low']
        #my_df['Close'] = my_df['Close']

        #my_df.to_csv(featpath+"/"+feat_filename)
        #print ("Saved "+str(series_no)+" succesfully")
    #except:
    #    print ('Error')
    #    pass
    
    
    return my_df

class RealTime ():
    def __init__ (self, instrument='USD_ZAR', modelname='', modelpath=''):
        if modelpath == '':
            self.modelpath = './models/weights'
        if modelname == '':
            self.modelname = 'trained_model_normalized_feat_stateless_fixed_timesteps_more_complex_512_2x_more_dropout'    
        self.daily_model = TradingModel (modelname=self.modelname,
                                         modelpath=self.modelpath)
        self.daily_model.loadModel ()
        self.intraday_model = TradingModel (modelname=self.modelname,
                                            modelpath=self.modelpath)
        self.intraday_model.loadModel ()
        self.instrument = instrument

    def loadCandlesToDataframe (self, granularity='D'): #D for daily or any granularity parameter of Oanda API
        candles = get_candles (instrument=self.instrument, 
                     granularity=granularity)

#my_candles = get_candles (granularity='D')
#print_candles (my_candles)

if False:
    instrument_list = ['EUR_USD', 'USD_ZAR', 'USD_JPY']
    
    timeframe_list = ['D', 'H4', 'H1']   

    for instrument in instrument_list:
        for timeframe in timeframe_list:
            start_date = dt.datetime (2001,1,1)
            end_date = end_date = start_date + relativedelta(years=1)    
            while start_date.year < 2017:
                ds = Dataset(lookback_window=1, ccy_pair=instrument, 
                             timeframe=timeframe)
                ds.initOnlineConfig ()
                try:
                    ds.loadSeriesOnline(bComputeFeatures=False, 
                                        bComputeLabels=False, 
                                        bSaveCandles=True,
                                        from_time=str(start_date),
                                        to_time=str(end_date))
                except:
                    print ('Error loading '+instrument+' '+timeframe+' '+str(start_date))
                end_date = end_date + relativedelta(years=1)
                start_date = start_date + relativedelta(years=1)

if False:
    instrument_list = ['USD_ZAR', 'USD_JPY']
    
    timeframe_list = ['H4']   

    for instrument in instrument_list:
        for timeframe in timeframe_list:
            start_date = dt.datetime (2001,1,1)
            end_date = end_date = start_date + relativedelta(years=1)    
            while start_date.year < 2017:
                ds = Dataset(lookback_window=1, ccy_pair=instrument, 
                             timeframe=timeframe)
                ds.initOnlineConfig ()
                
                try:
                    ds.loadCandles ()
                    ds.computeFeatures (bSaveFeatures=True,
                                        from_time=start_date + relativedelta(months=-4),
                                        to_time=end_date)
                except:
                    print ('Error loading '+instrument+' '+timeframe+' '+str(start_date))
                end_date = end_date + relativedelta(years=1)
                start_date = start_date + relativedelta(years=1)


if False:
    instrument_list = ['EUR_USD', 
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
    
    results = []
    res_inst = []
    for instrument in instrument_list[0:1]:
        if True:
        #try:
            my_candles = get_candles (instrument=instrument, 
                                      granularity='D', 
                                      from_time='2010-06-09 18:17:25', 
                                      to_time='2016-06-09 18:17:25', 
                                      default_n_periods=5000)
            #print ("Got "+str(len(my_candles))+" candles")
            
            #print ("Loading candles into dataset")
            df = loadCandlesIntoDataframe(my_candles)
            f_df = computeFeaturesOnTheFly(df)
            f_df = f_df.dropna ()
            sent = buildSentencesOnTheFly(f_df)
            X = buildSequencePatchesOnTheFly(sent, lookback_window=1)
            
            #-----------------------------------------
            ds = Dataset(lookback_window=1, n_features=X.shape[2])
            ds.feat_names_list = f_df.columns
            ds.X = X [600:,:,:]
        
            #labels = computeLabelsOnTheFly(f_df, target=0.05, stop=0.05)
            labels = computeLabelsOnTheFly(f_df, bVaryStopTarget =True)
            ds.y = np.zeros((len(labels), 3))
            for i in range(len(labels)):
                ds.y[i,1+np.int(labels[i])] = 1
            ds.y = ds.y[600:,:]
                     
            print (str(ds.X.shape))
                     
            rules_func_list = [rule_SimpleMomentum, 
                               rule_SimpleMomentumWithTrendingFilter,
                               rule_SimpleMomentumWithVIXFilter,
                               rule_SimpleMomentumWithVolFilter,
                               rule_ComplexMomentumWithTrendingFilter,
                               rule_complexRule,
                               rule_complexRuleWithRSI,
                               rule_veryComplex,
                               rule_complexRSIFading]
            
            results_elem = []
            for func in rules_func_list:
                try:
                    pred, [acc, neutral_ratio] = func (ds, verbose=True)
                    results_elem.append ([acc, neutral_ratio])
                        
                    print ('accuracy: '+str(acc))
                except:
                    pass
                
                    
        #except:
        #    print ('Error with instrument: '+instrument)
        else:
            results.append (results_elem)
            res_inst.append (instrument)
        
    
    acc_array = res[:,:,0]
    neutral_array = res[:,:,1]
    np.nanmean((acc_array-0.5) * (1.0-neutral_array), axis=0) / np.nanmean(1.0-neutral_array, axis=0)
    
    for i in range(29):
        print (res_inst [i]+ ', '+ str(res[i,-3,0] ) + ', '+str(res[i,-3,1]))