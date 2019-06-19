# -*- coding: utf-8 -*-

from Framework.Dataset.Dataset import *
from Config.const_and_paths import *
from Miscellaneous.my_utils import *

from copy import deepcopy
import ast
import time
import re

from View import printProgressBar

if V20_PATH not in sys.path:
    sys.path.append (V20_PATH)
if C_UTILS_PATH not in sys.path:
    sys.path.append (C_UTILS_PATH)
if CPP_UTILS_PATH not in sys.path:
    sys.path.append (CPP_UTILS_PATH)

from _C_arraytest import * #this name needs to be changed
#from cpp_utils_v2 import *  

def loadPeriodsOfInterest (periods_path=default_periods_path, 
                           periods_filename = default_periods_filename):
    f = open (os.path.join(periods_path, periods_filename), 'r')
    
    dict_str = f.read ()
    f.close ()
    
    periods_dict = ast.literal_eval(dict_str)
    
    return periods_dict

def getPeriodDict (period_name, periods = None,
        periods_path=default_periods_path, 
        periods_filename = default_periods_filename):
    if periods is None:
        periods = loadPeriodsOfInterest (periods_path=periods_path,
                        periods_filename = periods_filename)
    if 'Periods' in periods.keys ():
        periods_dict = periods['Periods']
    else:
        periods_dict = periods
        
    if period_name in periods_dict.keys ():
        return periods_dict[period_name]    
    
    return None

def savePeriodDict (period_name, period_dict, 
        periods = None,
        periods_path=default_periods_path, 
        periods_filename = default_periods_filename,
        bOverride=False):
    
    if periods is None:
        try:
            periods = loadPeriodsOfInterest (periods_path=periods_path,
                        periods_filename = periods_filename)
        
        
            if period_name in periods.keys () and not bOverride:
                print ('Period already saved')
                return
        except:
            print ('File did not exist previously')
            periods= {}
            periods = {}
        
        periods[period_name] = period_dict
        
        f = open (os.path.join(periods_path, periods_filename), 'w')
    
    
        f.write (str(periods))
    
        f.close ()
    

class DatasetHolder ():
    def __init__ (self, from_time='2015-01-01 00:00:00',
                  to_time='2016-01-01 00:00:00', instrument='EUR_USD', 
                  period_name = '', 
                  period_dict = {}, 
                  periods_path = default_periods_path,
                  periods_filename = default_periods_filename,
                  bLoadCandlesOnline = False,
                  bExceptions=True):
        
        self.bLoadCandlesOnline = bLoadCandlesOnline
        self.instrument = None
        self.init_instrument (instrument)
        
        self.ds_dict = {}

        self.bExceptions = bExceptions

        if str(type(from_time)) == "<type 'int'>":
            self.from_time = str (from_time) + '-01-01 00:00:00'
        else:
            self.from_time = from_time
            
        if str(type(to_time)) == "<type 'int'>":
            self.to_time = str (to_time) + '-12-31 23:59:59'
        else:        
            self.to_time = to_time
                
        self.period_name = period_name            
        if period_name != '' or len(period_dict) != 0:
            self.loadPeriodOfInterest (period_name = period_name, 
                                       period_dict = period_dict)
            
    def init_instrument (self, instrument=None):
        if instrument is not None:
            self.instrument = instrument
            
    def init_param (self, instrument, from_time, to_time):
        self.set_from_to_times (from_time, to_time)
        
        if instrument is not None:
            self.instrument = instrument        
        
    def set_from_to_times (self, from_time=None, to_time=None):
        if from_time is not None:
            if str(type(from_time)) == "<type 'int'>":
                self.from_time = str (from_time) + '-01-01 00:00:00'
            else:
                self.from_time = from_time
            
        if to_time is not None:
            if str(type(to_time)) == "<type 'int'>":
                self.to_time = str (to_time) + '-12-31 23:59:59'
            else:        
                self.to_time = to_time
        if self.from_time is None:
            self.from_time = '2000-01-01 00:00:00'
        if self.to_time is None:
            self.to_time = '2010-12-31 23:59:59'
    
    def loadMultiFrame (self, timeframe_list=['D', 'M15'],
                        ccy_pair_list = None, 
                        bComputeFeatures=[False, True],        
                        bComputeLabels=True,
                        bLoadFeatures=[True, False],
                        bLoadLabels=False):
        
        if ccy_pair_list is None:
            ccy_pair_list = [self.instrument]

        for i, ccy_pair in enumerate(ccy_pair_list):
            for k, timeframe in enumerate(timeframe_list):
                if ccy_pair + '_' + timeframe in self.ds_dict.keys ():
                    print ('Ccy pair already cached')
                    continue
                try:
                    #print ('Loading dataset ' + ccy_pair + '_' + timeframe)
                    if len (ccy_pair_list) >= 10:
                        printProgressBar(i+1,len (ccy_pair_list), suffix='Loading dataset: ' + str (ccy_pair), length=50)
                    ds = Dataset (lookback_window=1, ccy_pair=ccy_pair, 
                                  timeframe=timeframe, from_time=self.from_time,
                                  to_time=self.to_time,
                                  bLoadCandlesOnline = self.bLoadCandlesOnline)
                    ds.initOnlineConfig () 
                    ds.loadCandles ()
                    if bComputeFeatures[k]:
                        ds.computeFeatures ()
                    elif bLoadFeatures[k]:
                        ds.loadFeatures ()
                    
                    if bComputeLabels:
                        ds.computeLabels (bSaveLabels=False)
                    elif bLoadLabels:
                        ds.loadLabels ()
                    
                    self.ds_dict [ccy_pair +'_'+ timeframe] = ds
                except Exception as e:
                    print ('Error loading ' + ccy_pair + ' - ' + e.message)
                        
    def getLoadedInstruments (self):
        loaded_instruments = []
        for key in self.ds_dict.keys ():
            tf = TIMEFRAME_LIST[np.argmax([key.find (tf) for tf in TIMEFRAME_LIST])]            
            loaded_instruments.append (key[:-len(tf)-1])
        return list(set(loaded_instruments))
    
    def alignDataframes (self):
        min_datetime = self.from_time
        max_datetime = self.to_time
        
        for key in self.ds_dict.keys ():
            ds = self.ds_dict[key]
            if str (ds.f_df.index[0]) > min_datetime:
                min_datetime = str(ds.f_df.index[0])
            if str (ds.f_df.index[-1]) < max_datetime:
                max_datetime = str(ds.f_df.index[-1])
        
        self.from_time = min_datetime
        self.to_time = max_datetime
        for key in self.ds_dict.keys ():
            ds = self.ds_dict[key]
            ds.from_time = self.from_time
            ds.to_time = self.to_time
            ds.f_df = ds.f_df.loc[self.from_time:self.to_time]
            ds.buildXfromFeatDF ()
            ds.l_df = ds.l_df.loc[self.from_time:self.to_time]
            ds.buildYfromLabelDF ()
            
    def addDataset (self, ds):
        self.ds_dict[ds.ccy_pair + '_' + ds.timeframe] = deepcopy (ds)
        
    def appendTimeframesIntoOneDataset_slow (self, instrument=None, 
                                        higher_timeframe='D',
                                        lower_timeframe='H4', bComputeFeatures=False):
        self.init_instrument (instrument)
        
        ds_h = self.ds_dict[self.instrument+'_'+higher_timeframe]
        if bComputeFeatures:
            ds_h.computeFeatures ()
        ds_l = self.ds_dict[self.instrument+'_'+lower_timeframe]
        
        for i in range(len(ds_l.f_df)):
            ts = ds_l.f_df.index[i] - relativedelta(days=1)
            ts2 = ts - relativedelta(days=5)
            row_h = ds_h.f_df.loc[ts2:ts].ix[-1,:]
            if i == 0:
                for j, key in enumerate(row_h.keys ()):
                    idx_orig_feat = len(ds_l.f_df.columns)
                    ds_l.f_df[key + '_' + higher_timeframe] = np.zeros (len(ds_l.f_df))
            
            ds_l.f_df[idx_orig_feat:][i] = row_h
            if np.mod (i, 2000) == 0:
                print ('Mergin timeframes: ' + str(i) +'/' +str(len(ds_l.f_df)))
                
                
    def appendTimeframesIntoOneDataset (self, instrument=None, 
                                        higher_timeframe='D',
                                        lower_timeframe='H4',
                                        daily_delay = 0,
                                        bComputeFeatures=False,
                                        bConvolveCdl=True,
                                        conv_window=10,
                                        bCpp=False):
        self.init_instrument (instrument)
        
        ds_h = self.ds_dict[self.instrument+'_'+higher_timeframe]
            
        if bComputeFeatures:
            ds_h.computeFeatures ()
        ds_l = self.ds_dict[self.instrument+'_'+lower_timeframe]
        
        print ('DSH.append... f_df before: ' + str (ds_l.f_df.shape))

        if bConvolveCdl:
            fn = np.ones (conv_window)
            
            func_list = get_TA_CdL_Func_List ()
            
            for func in func_list:
                a = str (func)                
                cdl = a[a.find('CDL'):].split(' ')[0]
                cdl = re.sub('[^A-Za-z0-9]+', '', cdl)
                try:
                    if not ds_h.bConvolved:
                        ds_h.f_df[cdl] = np.convolve(ds_h.f_df[cdl], fn)[0:len(ds_h.f_df)]
                        ds_h.bConvolved = True
                    #ds_l.f_df[cdl] = np.convolve(ds_l.f_df[cdl], fn)[0:len(ds_l.f_df)]
                except:
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)
                
        if True:
            #removes duplicated columns
            for col in ds_l.f_df.columns:
                if col in [x + ('_' + ds_h.timeframe if not has_suffix(x) else '') for x in ds_h.f_df.columns]:
                    del ds_l.f_df[col]
                    
            cols = [col + ('_' + ds_l.timeframe if not has_suffix (col) else '') for col in ds_l.f_df.columns]
            cols2 = [col + ('_' + ds_h.timeframe if not has_suffix (col) else '') for col in ds_h.f_df.columns]
            
            #preparation to call C++ function
            if False:#bCpp:                
                start = time.time ()
                sys.path.append(CPP_UTILS_PATH)
                
                
                #C++ will be called from here
                a = StringVec ()
                map (a.append, [str(idx) for idx in ds_l.f_df.index])
                b = StringVec ()
                map (b.append, [str(idx) for idx in ds_h.f_df.index])
                #c = boost_multi_array_2d_from_numpy_array(np.array(ds_l.f_df))
                d = boost_multi_array_2d_from_numpy_array(np.array(ds_h.f_df))
                e = merge_frames (a, b, d, daily_delay)
                new_feat = nested_list_from_boost_multi_array_2d(e)
                data = np.hstack((ds_l.f_df[len(ds_l.f_df) - len (new_feat):], new_feat))
                for _ in [a, b, d, e, new_feat]:
                    del _
                
                ds_l.f_df = pd.DataFrame (data=data,
                                          index = ds_l.f_df.index[-np.shape(data)[0]:],
                                          columns = cols + cols2)
                print ('Ran C implementation successfully in ' + str(time.time () - start))
            else:                
                start = time.time ()
                #finds the indices to the slow dataframe that correspond
                #to the fast dataframe timestamps            
                idx_fast_str = ds_l.f_df.index
                idx_slow_str = ds_h.f_df.index
                fast_to_slow_indexing = np.zeros (len(idx_fast_str), int)
                
                i = 0
                j = 0
                while idx_slow_str[j] >= idx_fast_str[i]:
                    i+=1
                for j in range (len(idx_slow_str)-1):
                    try:
                        while (idx_fast_str[i] >= idx_slow_str[j]) and \
                         (idx_fast_str[i] < idx_slow_str[j+1]) and \
                            (i < len (idx_fast_str)):
                                fast_to_slow_indexing [i] = j
                                i += 1
                    except:
                        break
                #-----------------------------------------------------#
                #--------implements the delay-------------------------#
                fast_to_slow_indexing -= daily_delay
                offset = np.count_nonzero(fast_to_slow_indexing<0)            
                
                print ('Daily delay: ' + str (daily_delay))
                print ('offset: ' + str (offset))
                #new_feat = np.zeros ((len(ds_l.f_df) - offset, len(cols2)))
                
                #slow_feat_arr = np.array (ds_h.f_df)
                
                #length_fast = len(ds_l.f_df) - offset                
                #width_slow = slow_feat_arr.shape[1]
                
                #for i in range (length_fast):
                    #for j in range (width_fast):
                    #    new_feat_arr[i] [j] = fast_feat_arr [i + offset] [j]
                #    for j in range (width_slow):
                #        new_feat [i] [j] = slow_feat_arr [fast_to_slow_indexing [i + offset]] [j]
                #new_feat = slow_feat_arr [fast_to_slow_indexing [offset:], :]
                        
                data = np.hstack((np.array(ds_l.f_df[offset:]), 
                                  (np.array (ds_h.f_df)) [np.minimum(fast_to_slow_indexing [offset:],len (ds_h.f_df) - 1), :]))
                #creates resulting dataframe            
                ds_l.f_df = pd.DataFrame (data = data,
                                          index = idx_fast_str[offset:],
                                          columns = cols + cols2)
                try:
                    ds_l.l_df = ds_l.l_df.ix[offset:]
                except:
                    pass
                print ('Ran Python implementation in ' + str(time.time () - start))    
                print ('DSH.append... f_df after: ' + str (ds_l.f_df.shape))
        #except:
        #    print ('Both C++ and Python implementation failed')

    def buildSingleDataFrameFromDict (self, timeframe='H4'):
        samples = 0        
    
        for key in self.ds_dict.keys ():
            if self.ds_dict[key].timeframe == timeframe:
                samples += len(self.ds_dict[key].f_df)
                cols = self.ds_dict[key].f_df.columns
        
        idx = 0
        X = np.zeros ((samples, len (cols)))
        y = np.zeros (len(X))
        
        for key in self.ds_dict.keys ():
            if self.ds_dict[key].timeframe == timeframe:
                try:
                    X[idx:idx+len(self.ds_dict[key].f_df),:] = np.array (self.ds_dict[key].f_df)
                    y[idx:idx+len(self.ds_dict[key].f_df)] = np.array (self.ds_dict[key].l_df.Labels)
                    idx = idx+len(self.ds_dict[key].f_df)
                except:
                    print ('Error appending dataset of ' + key)
                    if self.bExceptions:
                        raise
                    else:
                        pass
        
        self.X = pd.DataFrame(data=X, columns=cols)
        self.y = y
        
    def loadOandaInstrumentListIntoOneDataset (self, bRandomize=False, instrument_list=[], daily_delay=2, bConvolveCdl=True, conv_window=10): 
        if len (instrument_list) == 0:
            instrument_list = [self.instrument]
        for ccy in instrument_list:
            try:
                self.init_instrument (ccy)
                self.loadMultiFrame ()
                
                if self.instrument + '_H4' in self.ds_dict.keys ():
                    ds = self.ds_dict[self.instrument + '_H4']
                    
                    if bRandomize:                
                        ds.randomizeCandles ()
                        ds.computeFeatures ()
                        ds.computeLabels ()
                        self.ds_dict[self.instrument + '_D'].buildCandlesFromLowerTimeframe(self.df)
                        self.ds_dict[self.instrument + '_D'].computeFeatures ()
                        
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
                    
                    idx_new_hilo = self.ds_dict[ccy + '_D'].getFeatIdx(feat='new_hilo_' + str(252) + '_' + str (20), 
                                          func=new_high_over_lookback_window, 
                                          args={'lookback_window':252, 
                                          'step_width':20,                      
                                          'feat':'Close'})
                    self.appendTimeframesIntoOneDataset (daily_delay = daily_delay,
                                                            bConvolveCdl=bConvolveCdl,
                                                            conv_window=conv_window)   
            except:
                try:
                    del self.ds_dict[ccy + '_D']
                except:
                    pass
                try:
                    del self.ds_dict[ccy + '_H4']
                except:
                    pass
                
                LogManager.get_logger ().error("Exception occurred", exc_info=True)
                
                if self.bExceptions:
                    raise
                else:
                    pass
                
        self.buildSingleDataFrameFromDict ()
        
    def savePeriodOfInterest (self, period_name='', 
                              periods_path = default_periods_path,
                              periods_filename = default_periods_filename,
                              bOverride=False):
        if period_name != '':
            self.period_name = period_name
        
        instrument_list = list(set([ds.ccy_pair for ds in self.ds_dict.values()]))
        timeframe_list = list(set([ds.timeframe for ds in self.ds_dict.values()]))
        
        
        period_dict = {'from_time': self.from_time,
          'instrument_list': instrument_list,
          'timeframes': timeframe_list,
          'to_time': self.to_time}
        
        savePeriodDict (period_name=self.period_name, period_dict=period_dict,             
            periods_path=periods_path, 
            periods_filename = periods_filename,
            bOverride=bOverride)
        
    def loadPeriodOfInterest (self, period_dict=None, period_name = '', 
                              periods_path=default_periods_path, 
                              periods_filename = default_periods_filename, 
                              from_time='2014-07-01 00:00:00', to_time='2015-07-31 23:59:59',
                              timeframes = ['D', 'H4'], instrument_list = None, bRandomize = False, bConvolveCdl=False, conv_window=10, daily_delay=2):
        
                
        if period_name != '':
            period_dict = getPeriodDict (period_name)
            
        if period_dict is not None:
            from_time = period_dict['from_time']
            timeframes = period_dict['timeframes']
            instrument_list = period_dict['instrument_list']
            
            if type (instrument_list) == str:
                instrument_list = [instrument_list]

        if instrument_list is None:
            instrument_list = [self.instrument]
        
        set_from_to_times (self, from_time = from_time, to_time = to_time)
        self.loadOandaInstrumentListIntoOneDataset (bRandomize=bRandomize, 
                                                    instrument_list=instrument_list, 
                                                    daily_delay=daily_delay, 
                                                    bConvolveCdl=bConvolveCdl, 
                                                    conv_window=conv_window)
        

if False:
    import pandas as pd
    import numpy as np
    
    dsh = DatasetHolder (from_time=2014, to_time=2014, bExceptions=False)
    dsh.loadMultiFrame(timeframe_list=['H4'], 
                       ccy_pair_list=['USD_CAD', 'AUD_USD', 'USD_ZAR', 'USD_NOK', 'USD_TRY'])
    
    idx = pd.core.indexes.datetimes.DatetimeIndex ([])
    keys = dsh.ds_dict.keys ()
    
    for key in keys:
        if key in dsh.ds_dict.keys ():
            length = len(dsh.ds_dict[key].df)
            if len(dsh.ds_dict[key].df.dropna()) < 0.75 * length:            
                del dsh.ds_dict[key]
            else:
                idx = pd.core.indexes.datetimes.DatetimeIndex(set(idx.append(dsh.ds_dict[key].df.index)))
    
    idx = idx.sort_values(ascending=True)
    window = 360
    
    l1 = []
    l2 = []
    l3 = []
    for j in range (window, len(idx)):
        try:
            print ('Processing ' + str(j) +' out of ' + str(len(idx) - window))
            arr = np.zeros ((len(dsh.ds_dict.keys ()) - 0, window))
            for i, key in enumerate(dsh.ds_dict.keys ()[0:]):
                arr[i,:] = np.array (dsh.ds_dict[key].df.ix[idx[j-window:j]].Close)
                print (key + ', ' + str(dsh.ds_dict[key].df.ix[idx[j-window:j]].Close.isnull().sum()))
                
            new_df = pd.core.frame.DataFrame(arr.transpose())
            new_df.dropna(inplace=True)
            
            ret_arr = np.nan_to_num(np.log(np.array(new_df)[1:,:] / np.array(new_df)[0:-1,:]).transpose ())
            C = np.corrcoef(ret_arr)
            S,V, D = np.linalg.svd(C)
            PC = np.dot(ret_arr.transpose(), S).transpose ()
            l1.append (V)
            l2.append(PC)
            
            corr = []
            for i, key in enumerate(dsh.ds_dict.keys ()):
                corr_elem = []
                for n in range (2):
                    corr_elem.append (np.corrcoef(ret_arr[i,:], PC[n])[0,1])
                corr.append(corr_elem)
            l3.append(corr)
        except:
            pass
        
if False:
    for j, key in enumerate(dsh.ds_dict.keys ()):
        fig = plt.figure ()
        plt.plot(np.cumprod(1+ret_arr[j,:]), label=key)
        for i in range(0, PC.shape[0]):
            plt.plot(np.cumprod(1+PC[i,:]), label=str(i))
            plt.legend(loc='best')
            
    for j, key in enumerate(dsh.ds_dict.keys ()):
        for i in range(0, PC.shape[0]):
            corr_mat = np.corrcoef(ret_arr[j,:], PC[i,:])
            print ('Correlation ' + key + ' x PC-' + str(i) +': ' + str(corr_mat[0,1]))
        print ('')

if False:
    from johansen import Johansen
    
    for i in range (ret_arr.shape[0]):
        for j in range(1):
            arr = np.zeros((PC.shape[1], 2))        
            arr[:,0] = np.exp(np.cumsum(PC[j,:]))
            arr[:,1] = np.exp(np.cumsum(ret_arr[i,:]))
                    
            prices_df = pd.core.frame.DataFrame(arr)        
            prices_df.dropna(inplace=True)        
            x = prices_df.as_matrix()        
            x_centered = x - np.mean(x, axis=0)
            
            johansen = Johansen(x_centered, model=2, significance_level=0)
            eigenvectors, r = johansen.johansen()
            
            vec = eigenvectors[:, 0]
            vec_max = np.max(np.abs(vec))
            vec = vec / vec_max
            
            portfolio_insample = np.dot(x, vec)
            in_sample = np.dot(x, vec)
            mean = np.mean(in_sample)
            std = np.std(in_sample)
            fig = plt.figure ()
            plt.title (str(i) + ' ' + str(j))
            plt.plot(portfolio_insample, '-')
    
            plt.axhline(y=mean - 2 * std, color='r', ls='--', alpha=.5)
            plt.axhline(y=mean, color='r', ls='--', alpha=.5)
            plt.axhline(y=mean + 2 * std, color='r', ls='--', alpha=.5)
            
            plt.show()

    