# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:03:01 2016

@author: Joanna
"""

import pandas
import numpy as np
from matplotlib import pyplot as plt
from Trading.indicators import *
from Trading.Technical.TrendlineFeaturesExtractor import *

def p2f(x):
    return float(x.strip('%'))/100

featpath = './datasets/Fx/Featured/NotNormalizedNoVolume/NewFeatures'
parsedpath = './datasets/Fx/Parsed'
labelpath = './datasets/Fx/Labeled'

no_series = 116
series_no = 17
show_last = 1500
lookback_window = 252

#TODO
#1/ normalize features
func_list = get_TA_CdL_Func_List ()

print ("---------Extracting Features----------------------")

for series_no in range(18,116,1):
    
    parsed_filename = 'ccy_hist_ext_'+str(series_no)+'.txt'
    print ("Raw data file: " + parsed_filename)
    feat_filename = 'not_normalized_new_feat_'+str(series_no)+'.csv'
     
    
    
    #-------------------computes the features----------------------------------#
    #if True:
    try:
        my_df = pandas.read_csv(parsedpath+'/'+parsed_filename, names = ["Date", "Close", "Open", "High", "Low", "Change"], converters={'Change':p2f})
        #my_df = pandas.read_csv(parsedpath+'/'+parsed_filename)
        my_df['Date'] = pandas.to_datetime(my_df['Date'], infer_datetime_format=True)
        my_df.index = my_df['Date']
        my_df = my_df.sort(columns='Date', ascending=True)
        #-----moving averages----------------#
        my_df['ma_21_close'] = my_df['Close'].rolling(window=21).mean()
        my_df['ma_50_close'] = my_df['Close'].rolling(window=50).mean()
        my_df['ma_100_close'] = my_df['Close'].rolling(window=100).mean()
        my_df['ma_200_close'] = my_df['Close'].rolling(window=200).mean()
        
        #_____hist vol_______
        my_df['hist_vol_2wk_close'] = my_df['Change'].rolling(window=10).std()
        my_df['hist_vol_1m_close'] = my_df['Change'].rolling(window=22).std()
        my_df['hist_vol_3m_close'] = my_df['Change'].rolling(window=66).std()
        my_df['hist_vol_6m_close'] = my_df['Change'].rolling(window=126).std()
        my_df['hist_vol_1y_close'] = my_df['Change'].rolling(window=252).std()
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
        
        for func in func_list:
            func_name = str(func)[19:-2]
            my_df[func_name] = func (open_a, high, low,
                                        close) / 100

        #compute moving average crossovers
        #compute new features based on highs-lows
        x = np.array(my_df['Close'])
        x_rsi = np.array (my_df['RSI'])
        for relevant_threshold in [5,10,20,30]:
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
        
            seg_length = 600
            
            print ("Series no: "+str(series_no)+' thres: '+str(relevant_threshold))
            
            for i in range (0, len(x)):                
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
            
            #------------ratios of close vs moving averages normalized by vol-----------------------#
            my_df['close_over_21d_ma_normbyvol'] = (close - ma_21) / hist_vol_3m
            my_df['close_over_50d_ma_normbyvol'] = (close - ma_50) / hist_vol_3m
            my_df['close_over_100d_ma_normbyvol'] = (close - ma_100) / hist_vol_3m
            my_df['close_over_200d_ma_normbyvol'] = (close - ma_200) / hist_vol_3m
    
            #-----------ratios of moving averages normalized by vol-------------------------------#
            my_df['21d_over_100d_ma_normbyvol'] = (ma_21 - ma_100) / hist_vol_3m
            my_df['21d_over_200d_ma_normbyvol'] = (ma_21 - ma_200) / hist_vol_3m
            my_df['50d_over_200d_ma_normbyvol'] = (ma_50 - ma_200) / hist_vol_3m

            my_df['dist_standing_high_normbyvol'+str(relevant_threshold)] = dist_standing_high_list / hist_vol_3m
            my_df['dist_standing_low_normbyvol'+str(relevant_threshold)] = dist_standing_low_list / hist_vol_3m
            my_df['dist_relevant_low_normbyvol'+str(relevant_threshold)] = dist_relevant_low_list / hist_vol_3m
            my_df['dist_relevant_high_normbyvol'+str(relevant_threshold)] = dist_relevant_high_list / hist_vol_3m
        
        #normalize the candle prices
        my_df['Open'] = my_df['Open']
        my_df['High'] = my_df['High']
        my_df['Low'] = my_df['Low']
        my_df['Close'] = my_df['Close']

        my_df.to_csv(featpath+"/"+feat_filename)
        print ("Saved "+str(series_no)+" succesfully")
    except:
        print ('Error')
        pass
    

 