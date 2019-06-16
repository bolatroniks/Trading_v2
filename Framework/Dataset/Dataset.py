# -*- coding: utf-8 -*-

from Framework.Reporting.Logging.LogManager import LogManager

try:
    from Framework.Strategy.Rules import *
except:
    LogManager.get_logger ().error("Exception occurred", exc_info=True)

import sys
v20_path = v20_path = '../../../oanda/src'

if v20_path not in sys.path:
    sys.path.append (v20_path)

sys.path.append ('~/.v20.conf')

from Framework.Dataset.RealTime.Realtime import *
from Framework.FeatureExtractors.Technical.indicators import get_TA_CdL_Func_List
from Framework.Strategy.Rules.Rule import *
from Framework.Reporting.Logging.LogManager import LogManager

import os

func_list = get_TA_CdL_Func_List ()

#checks if a column name has a suffix like _D, _H4, ...
has_suffix = lambda col: col[(-col[::-1].find ('_') if col[::-1].find ('_') > 0 else 0):] in TF_LIST


class FailedLoadingSeriesOnline(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


func_list = get_TA_CdL_Func_List()


def set_from_to_times(obj, from_time=None, to_time=None):
    if from_time is not None:
        if str(type(from_time)) == "<type 'int'>":
            obj.from_time = str(from_time) + '-01-01 00:00:00'
        else:
            obj.from_time = from_time

    if to_time is not None:
        if str(type(to_time)) == "<type 'int'>":
            obj.to_time = str(to_time) + '-12-31 23:59:59'
        else:
            obj.to_time = to_time
    if obj.from_time is None:
        obj.from_time = '2000-01-01 00:00:00'
    if obj.to_time is None:
        obj.to_time = '2010-12-31 23:59:59'


class Dataset():
    def __init__(self,
                 ccy_pair=None,  # the instrument to be loaded
                 timeframe='D',  # daily
                 from_time=None,  # starting date for which to load data
                 to_time=None,  # end date for which to load data
                 bConvolveCdl=True,  # candlestick patterns are a vector of 0s and 1s (or 100s)
                 # this parameter tells whether to convolve these features with a window function

                 # dataframes - core of this class
                 df=None,  # contains the parsed data - candles dataframe
                 f_df=None,  # contains the featured dataframe
                 l_df=None,  # contains the labels dataframe
                 p_df=None,  # contains the predictions dataframe

                 # ------Parameters for supervised learning---------#
                 lookback_window=1,  # sequence size for LSTM
                 n_features=75,  # width of features vector
                 no_signals=3,  # number of categories to predict
                 cv_set_size=1000,  # length of cross-validation set
                 test_set_size=1000,  # length of testing set
                 bZeroMA_train=False,  # whether to normalize returns so that average is zero
                 bZeroMA_cv=False,  # whether to normalize returns so that average is zero
                 bZeroMA_test=False,  # whether to normalize returns so that average is zero
                 period_ahead=0,  # try to predict returns n periods ahead
                 min_period_ahead=15,  # try to predict returns n periods ahead
                 max_period_ahead=27,  # try to predict returns n periods ahead
                 bCenter_y=True,  # used for relabeling dataset
                 bCenter_cv_y=False,  # used for relabeling dataset
                 bCenter_test_y=False,  # used for relabeling dataset
                 ret_type='log',  # used for relabeling dataset

                 # -----------paths, prefixes and suffixes-------------#
                 featpath=FEATURES_DEFAULT_PATH,
                 parsedpath=PARSED_MKT_DATA_DEFAULT_PATH,
                 labelpath=LABELS_DEFAULT_PATH,
                 predpath=PREDS_DEFAUL_PATH,
                 pca_feat_path= os.path.join (PCA_DEFAULT_PATH, 'VAR'),
                 feat_filename_prefix='ccy_hist_normalized_feat_',
                 label_filename_prefix='ccy_hist_feat_',
                 pred_filename_prefix='ccy_hist_feat_',
                 feat_filename_suffix='.csv',
                 label_filename_suffix='.csv',
                 # ----------------------------------------------------#

                 bOnlineConfig=True,  # True, uses Oanda settings
                 bLoadCandlesOnline=False,  # if False, loads candles from disk
                 feat_names=None,
                 series_no=None,  # index for a different dataset

                 # -----------parameters used to compute variable target and stop losses---------------#
                 min_stop=None,  # parameter used to compute variable targets
                 target_multiple=None,
                 vol_denominator=None,  # parameter used to compute variable targets
                 bVaryStopTarget=True,  # boolean to decide whether to compute variable targets
                 stop_fn=None,  # function to compute variable stop losses
                 target_fn=None,  # function to compute variable targets
                 # ------------------------------------------------------------------------------------#

                 high_low_feat_window = 500,
                 bLoadOnlyTrainset=False,  # loads only train set to save memory
                 bNormalize=False,  # normalize features
                 bRelabel=False,  # re-computes labels
                 bVerbose=False  # prints stuff during execution
                 ):

        self.total_no_series = 1000
        self.last = 2000  # max number of samples to be used from each series for training
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.no_signals = no_signals

        self.cv_set_size = cv_set_size
        self.test_set_size = test_set_size
        self.bZeroMA_train = bZeroMA_train
        self.bZeroMA_cv = bZeroMA_cv
        self.bZeroMA_test = bZeroMA_test

        self.featpath = featpath
        self.parsedpath = parsedpath
        self.labelpath = labelpath
        self.predpath = predpath
        self.pca_feat_path = pca_feat_path

        self.parsed_filename_prefix = ''
        self.feat_filename_prefix = feat_filename_prefix
        self.label_filename_prefix = label_filename_prefix
        self.pred_filename_prefix = pred_filename_prefix

        self.pred_filename_suffix = '.csv'
        self.parsed_filename_suffix = '.csv'
        self.feat_filename_suffix = feat_filename_suffix
        self.label_filename_suffix = label_filename_suffix

        if bOnlineConfig:
            self.initOnlineConfig()
        self.bLoadCandlesOnline = bLoadCandlesOnline

        self.feat_names = None  # used to pass feature names to read files without headers
        self.feat_names_list = []  # used to store the feature names

        self.X = []
        self.y = []
        self.cv_X = []
        self.cv_y = []
        self.test_X = []
        self.test_y = []
        
        self.bConvolved = False

        self.period_ahead = period_ahead  # used for re-labelling dataset, X[t,-1,0] vs X[t-period_ahead,-1,0]
        self.min_period_ahead = min_period_ahead
        self.max_period_ahead = max_period_ahead
        self.bCenter_y = bCenter_y  # used for relabeling dataset: subtracts average from returns
        self.bCenter_cv_y = bCenter_cv_y  # used for relabeling dataset
        self.bCenter_test_y = bCenter_test_y  # used for relabeling dataset
        self.ret_type = ret_type  # used for relabeling dataset: how to compute the return, x[t]/x[t-h] - 1 or log(x[t]/x[t-h])

        # -------------normalization indices-------------#
        # -------------not used for Oanda dataset--------#
        self.mu_sigma_list = [0, 1, 2, 3, 5, 6, 7, 8, 23, 24, 25, 34, 36, 46, 47, 48, 49, 50, 51, 56, 58, 59, 72, 79]
        self.by100_list = [16, 17, 19, 20, 21, 27, 28, 29, 30, 31, 52, 53, 54, 55, 60, 61, 62, 65, 66, 68, 71, 73, 74,
                           75, 76, 78, 80]
        self.volume_feat_list = []
        # -----------------------------------------------#

        self.series_no = series_no
        self.ccy_pair = ccy_pair
        self.timeframe = timeframe

        self.min_stop = None
        self.vol_denominator = None
        self.bVaryStopTarget = None
        self.stop_fn = None
        self.target_fn = None
        self.target_multiple = None

        self.set_label_parameters(bVaryStopTarget=bVaryStopTarget,
                                  min_stop=min_stop,
                                  vol_denominator=vol_denominator,
                                  target_multiple=target_multiple,
                                  stop_fn=stop_fn,
                                  target_fn=target_fn)

        self.bLoadOnlyTrainset = bLoadOnlyTrainset
        self.bNormalize = bNormalize
        self.bRelabel = bRelabel
        self.bConvolveCdl = bConvolveCdl

        if series_no is not None:
            self.loadSeriesByNo(self.series_no,
                                self.bLoadOnlyTrainset,
                                self.bNormalize,
                                self.bRelabel,
                                self.bConvolveCdl)

        self.from_time = from_time
        self.to_time = to_time
        self.ccy_pair = ccy_pair
        self.timeframe = timeframe
        self.set_from_to_times(from_time, to_time)

        self.df = df
        self.f_df = f_df
        self.l_df = l_df
        self.p_df = p_df

        self.bVerbose = bVerbose
        
        self.high_low_feat_window = high_low_feat_window

    def init_param(self, instrument, timeframe, from_time, to_time):
        self.set_from_to_times(from_time, to_time)

        if instrument is not None:
            self.ccy_pair = instrument
        if timeframe is not None:
            self.timeframe = timeframe

    def set_from_to_times(self, from_time=None, to_time=None):
        if from_time is not None:
            if str(type(from_time)) == "<type 'int'>":
                self.from_time = str(from_time) + '-01-01 00:00:00'
            else:
                self.from_time = from_time

        if to_time is not None:
            if str(type(to_time)) == "<type 'int'>":
                self.to_time = str(to_time) + '-12-31 23:59:59'
            else:
                self.to_time = to_time
        if self.from_time is None:
            self.from_time = '2000-01-01 00:00:00'
        if self.to_time is None:
            self.to_time = '2013-12-31 23:59:59'

    def initOnlineConfig(self):
        self.parsedpath = PARSED_MKT_DATA_DEFAULT_PATH
        self.labelpath = LABELS_DEFAULT_PATH
        self.featpath = FEATURES_DEFAULT_PATH
        self.predpath = PREDS_DEFAUL_PATH
        self.label_filename_prefix = 'Oanda_'
        self.feat_filename_prefix = 'Oanda_'
        self.parsed_filename_prefix = 'Oanda_'
        self.pred_filename_prefix = 'Oanda_'

        return self

    def loadSeriesOnline(self, instrument=None,
                         timeframe=None,
                         from_time=None,
                         to_time=None,
                         default_n_periods=5000,
                         bComputeFeatures=False,
                         bComputeLabels=False, target=0.025, stop=0.025, bVaryStopTarget=True,
                         bSaveCandles=False, candles_filename='',
                         bSaveFeatures=False, feat_filename='',
                         bSaveLabels=False, labels_filename=''):

        self.init_param(instrument, timeframe, from_time, to_time)

        my_candles = get_candles(instrument=self.ccy_pair,
                                 granularity=self.timeframe,
                                 from_time=self.from_time,
                                 to_time=self.to_time,
                                 default_n_periods=default_n_periods)
        if my_candles is None:
            raise FailedLoadingSeriesOnline("Candles list is None")
        if len(my_candles) > 0:

            self.df = loadCandlesIntoDataframe(my_candles)
            if bSaveCandles:
                self.saveCandles()

            if bComputeFeatures:
                self.computeFeatures(bSaveFeatures=bSaveFeatures)

                if bComputeLabels:
                    self.computeLabels(bSaveLabels=bSaveLabels,
                                       target=target, stop=stop,
                                       bVaryStopTarget=bVaryStopTarget)
        else:
            raise FailedLoadingSeriesOnline("Candles empty")
            
        return self

    # This method is used to test a strategy on random data
    # If the return is positive and consistent
    # it means look ahead bias
    def randomizeCandles(self):
        self.df['Vol'] = self.df['Change'].rolling(window=22).std()

        new_change = np.zeros(len(self.df))
        for i in range(len(new_change)):
            new_change[i] = np.random.normal(0.0, self.df.Vol[i], 1)[0]

        ratio_open_close = self.df.Open / self.df.Close
        ratio_high_close = self.df.High / self.df.Close
        ratio_low_close = self.df.Low / self.df.Close

        self.df.Change = new_change
        self.df.Close = self.df.Close[0] * np.cumprod(1 + self.df.Change)
        self.df.Open = self.df.Close * ratio_open_close
        self.df.High = self.df.Close * ratio_high_close
        self.df.Low = self.df.Close * ratio_low_close
        try:
            self.f_df = None
            self.l_df = None
            self.p_df = None
        except Exception as e:
            LogManager.get_logger ().error("Exception occurred", exc_info=True)

    # uses fast timeframe data to build slower ones
    def buildCandlesFromLowerTimeframe(self, ds = None, df=None, lower_timeframe='M15',
                                       hour_daily_open = 9, hour_daily_close = 17):
        cols = ['Open', 'High', 'Low', 'Close', 'Change', 'Volume']

        if ds is not None:
            df = ds.df
            lower_timeframe = ds.timeframe            
        elif df is None:
            ds_aux = Dataset(ccy_pair=self.ccy_pair, timeframe=lower_timeframe,
                             from_time=self.from_time, to_time=self.to_time,
                             bLoadCandlesOnline=self.bLoadCandlesOnline)
            ds_aux.loadCandles()
            df = ds_aux.df
        
        if self.timeframe == 'H1':
            idx = np.array([i for i in range(len(df.index)) if df.index[i].minute == 0])
            if lower_timeframe == 'M15':
                ratio = 4
        if self.timeframe == 'H4':
            idx = np.array([i for i in range(len(df.index)) if df.index[i].minute == 0 and df.index[i].hour in [1, 5, 9, 13, 17, 21]])
            if lower_timeframe == 'H1':
                ratio = 4
            elif lower_timeframe == 'M15':
                ratio = 16
        if self.timeframe == 'D':
            idx = np.array([i for i in range(len(df.index)) if
                            (df.index[i].minute == 0) and (df.index[i].hour == 17)])
            
            if lower_timeframe == 'H4':
                ratio = 2
            elif lower_timeframe == 'H1':
                ratio = 8
            elif lower_timeframe == 'M15':
                ratio = 32
        if self.timeframe == 'W':
            idx = np.array([i for i in range(len(df.index)) if
                            (df.index[i].minute == 0) and (df.index[i].hour == 17) and df.index[i].dayofweek == 4])
            if lower_timeframe == 'D':
                ratio = 5
            if lower_timeframe == 'H4':
                ratio = 4 * 6 + 2
            elif lower_timeframe == 'H1':
                ratio = ( 4 * 6 + 2 ) * 4
            elif lower_timeframe == 'M15':
                ratio = ( 4 * 6 + 2 ) * 16

        arr = np.array(df)

        #stars from 1: because of the situation where idx[0] - ratio < 0
        data = np.zeros((len(idx), len(cols)), float)
        data[1:, 0] = arr[idx[1:] - ratio, 0]  # open
        data[1:, 1] = [np.max(arr[idx[i] - ratio:idx[i], 1])
                      for i in range(1, len(idx))]  # high
        data[1:, 2] = [np.min(arr[idx[i] - ratio:idx[i], 2])
                      for i in range(1, len(idx))]  # low
        data[1:, 3] = arr[idx[1:], 3]  # close
        data[2:, 4] = np.log(data[2:, 3] / data[1:-1, 3])  # change
        data[1:, 5] = [np.sum(arr[idx[i] - ratio:idx[i], 5])
                      for i in range(1, len(idx))]  # volume
        
        #'fixes' the situation where idx[0] - ratio < 0
        #code looks uglier, but better fix it like this to avoid overhead
        if idx[0] - ratio - 0:
            first_idx = 0
        else:
            first_idx = idx[0] - ratio
        data[0, 0] = arr[first_idx, 0]  # open
        data[0, 1] = np.max(arr[first_idx:idx[0], 1]) #high                      
        data[0, 2] = np.min(arr[first_idx:idx[0], 1]) # low
        data[0, 3] = arr[idx[0], 3]  # close
        data[1, 4] = np.log(data[1, 3] / data[0, 3])  # change
        data[0, 5] = np.sum(arr[first_idx:idx[0], 5]) # volume
        #end of fudge

        self.df = pandas.core.frame.DataFrame(data=data, index=df.index[idx], columns=cols)

        try:
            self.f_df = None
            self.l_df = None
            self.p_df = None
        except Exception as e:
            LogManager.get_logger ().error("Exception occurred", exc_info=True)

        if False:
            dates = []
            highs = []
            lows = []
            opens = []
            closes = []
            for i in range(0, len(df) - 6, 6):
                dates.append(df.index[i + 6])
                opens.append(df.loc[df.index[i]:df.index[i + 6]].Open[0])
                closes.append(df.loc[df.index[i]:df.index[i + 6]].Close[-1])
                highs.append(np.max(df.loc[df.index[i]:df.index[i + 6]].High))
                lows.append(np.min(df.loc[df.index[i]:df.index[i + 6]].Low))

            self.df = None
            self.df = pandas.core.frame.DataFrame()
            self.df['Date'] = dates
            self.df['Open'] = opens
            self.df['Close'] = closes
            self.df['High'] = highs
            self.df['Low'] = lows
            self.df['Change'] = closes
            for i in range(1, len(self.df)):
                self.df['Change'][i] = self.df.Close[i] / self.df.Close[i - 1] - 1.0
            self.df.index = self.df.Date
            del self.df['Date']
            try:
                self.f_df = None
                self.l_df = None
                self.p_df = None
            except:
                LogManager.get_logger ().error("Exception occurred", exc_info=True)

        return self.df

    def buildCandlesFromLowerTimeframe_v2(self, df):
        highs = [0]
        lows = [0]
        opens = [0]
        closes = [0]
        for i in range(1, len(self.df)):
            # dates.append (df.index[i])
            opens.append(df.loc[self.df.index[i - 1]:self.df.index[i]].Open[0])
            closes.append(df.loc[self.df.index[i - 1]:self.df.index[i]].Close[-1])
            highs.append(np.max(df.loc[self.df.index[i - 1]:self.df.index[i]].High))
            lows.append(np.min(df.loc[self.df.index[i - 1]:self.df.index[i]].Low))

        # self.df = None
        # self.df = pandas.core.frame.DataFrame ()
        # self.df['Date'] = dates
        self.df['Open'] = opens
        self.df['Close'] = closes
        self.df['High'] = highs
        self.df['Low'] = lows
        self.df['Change'] = closes
        for i in range(1, len(self.df)):
            self.df['Change'][i] = self.df.Close[i] / self.df.Close[i - 1] - 1.0
        # self.df.index = self.df.Date
        # del self.df['Date']
        try:
            self.f_df = None
            self.l_df = None
            self.p_df = None
        except:
            LogManager.get_logger ().error("Exception occurred", exc_info=True)

    # sets l_df.Labels to one of the many labels possibilities
    def get_active_labels(self):
        try:
            label_key = 'Labels' + get_labels_suffix(self.bVaryStopTarget,
                                                     self.min_stop,
                                                     self.vol_denominator,
                                                     self.target_multiple,
                                                     self.stop_fn,
                                                     self.target_fn)
            self.l_df['Labels'] = self.l_df[label_key]
            return self.l_df.Labels
        except KeyError as error_str:
            print ('Key not found: ' + str(label_key))
            if 'Labels' in self.l_df.columns:
                print ('Using default Labels column')
            else:
                raise error_str

    def set_label_parameters(self, bVaryStopTarget=None,
                             min_stop=None,
                             vol_denominator=None,
                             target_multiple=None,
                             stop_fn=None,
                             target_fn=None):
        min_stop_dict = {'W': 0.05,
                        'D': 0.02,
                         'H4': 0.01,
                         'H1': 0.007,
                         'M15': 0.005}

        vol_denom_dict = {
                            'W': 1,
                            'D': 5,
                          'H4': 10,
                          'H1': 20,
                          'M15': 30}

        if bVaryStopTarget is not None:
            self.bVaryStopTarget = bVaryStopTarget
        elif self.bVaryStopTarget is None:
            self.bVaryStopTarget = True

        if min_stop is not None:
            self.min_stop = min_stop
        elif self.min_stop is None:
            self.min_stop = min_stop_dict[self.timeframe]

        if vol_denominator is not None:
            self.vol_denominator = vol_denominator
        elif self.vol_denominator is None:
            self.vol_denominator = vol_denom_dict[self.timeframe]

        if target_multiple is not None:
            self.target_multiple = target_multiple
        elif self.target_multiple is None:
            self.target_multiple = 1.0

        if stop_fn is not None:
            self.stop_fn = stop_fn
        if target_fn is not None:
            self.target_fn = target_fn
            
        return self

    def computeLabels(self, bSaveLabels=False,
                      from_time=None,
                      to_time=None,
                      bVaryStopTarget=True,
                      stop_fn=None, target_fn=None,
                      target_multiple=None,
                      min_stop=None,
                      vol_denominator=None,
                      bBuildY=False  # for supervised learning
                      ):

        self.set_from_to_times(from_time, to_time)
        self.set_label_parameters(bVaryStopTarget=bVaryStopTarget,
                                  min_stop=min_stop,
                                  vol_denominator=vol_denominator,
                                  target_multiple=target_multiple,
                                  stop_fn=stop_fn,
                                  target_fn=target_fn)

        if self.f_df is None:
            try:
                self.loadFeatures()
            except:
                print ('Features dataset not ready yet')
                return

        labels_dict = computeLabelsOnTheFly(self.f_df,
                                            tf_suffix=('' if 'Close' in self.f_df.columns else '_' + self.timeframe),
                                            bVaryStopTarget=self.bVaryStopTarget,
                                            min_stop=self.min_stop,
                                            vol_denominator=self.vol_denominator,
                                            stop_fn=self.stop_fn,
                                            target_fn=self.target_fn,
                                            target_multiple=self.target_multiple)
        # .loc[self.from_time:self.to_time]
        self.l_df = deepcopy(self.f_df.ix[:, 0:6])
        for key in labels_dict.keys():
            self.l_df[key] = labels_dict[key]
            
        close_col = ('Close' if 'Close' in self.f_df.columns else 'Close_' + self.timeframe)
        vol_col = ('hist_vol_1m_close' if 'hist_vol_1m_close' in self.f_df.columns else 'hist_vol_1m_close_' + self.timeframe)
            
        self.l_df ['ret_10_periods'] = np.log(self.l_df.shift(-10)[close_col] / self.l_df[close_col]) / self.f_df[vol_col]
        self.l_df ['ret_25_periods'] = np.log(self.l_df.shift(-35)[close_col] / self.l_df[close_col]) / self.f_df[vol_col]
        self.l_df ['ret_50_periods'] = np.log(self.l_df.shift(-50)[close_col] / self.l_df[close_col]) / self.f_df[vol_col]

        self.get_active_labels()

        if bBuildY:
            self.buildYfromLabelDF()

        if bSaveLabels:
            self.saveLabels()
            
        return self

    def computePredictions(self, bSavePredictions=False,
                           bRemoveSerialPredictions=False,
                           serial_gap=10,
                           bRemoveLastPredictions=True,
                           last_window=None,
                           rule=None):
        # should have some checks here
        
        if last_window is None:
            if self.timeframe == 'D':
                last_window = 20
            elif self.timeframe == 'H4':
                last_window = 20 * 6
            elif self.timeframe == 'H1':
                last_window = 20 * 24
            elif self.timeframe == 'M15':
                last_window = 20 * 24 * 4

        if rule is not None:            
            pred, dummy = rule.func(self)
            pred[-np.minimum(len(pred), last_window):] = NEUTRAL_SIGNAL
            self.set_predictions(pred)
        else:
            self.set_predictions (NEUTRAL_SIGNAL * np.ones (len (self.f_df)))
        

        if bRemoveSerialPredictions:
            self.removeSerialPredictions(serial_gap)

        if bSavePredictions:
            self.savePredictions(rule.name)
            
        return self

    def set_predictions(self, pred = None):
        self.p_df = deepcopy(self.f_df.ix[:, 0:6])
        
        if pred is not None:
            self.p_df['Predictions'] = deepcopy(pred)
        else:
            self.p_df['Predictions'] = np.ones (len (self.p_df)) * NEUTRAL_SIGNAL
        
        return self

    def removeSerialPredictions(self, serial_gap):
        removed_count = 0

        preds = np.array(self.p_df.Predictions)

        for i in range(serial_gap, len(preds), 1):
            if np.sum((preds[i - serial_gap:i] - NEUTRAL_SIGNAL) ** 2) > 0.0:
                # print ('Serial prediction removed')
                removed_count += 1
                preds[i] = NEUTRAL_SIGNAL
        self.p_df.Predictions = preds

        if self.bVerbose:
            print ('Serial prediction removed: ' + str(removed_count))
            
        return self

    def evaluatePredictions(self):
        labels = self.l_df.Labels
        if np.min(labels) != SHORT_SIGNAL and np.max(labels) != LONG_SIGNAL:
            labels -= 1
        predictions = self.p_df.Predictions
        if np.min(predictions) != SHORT_SIGNAL and np.max(predictions) != LONG_SIGNAL:
            predictions -= 1
        non_neutral = len(predictions[predictions != NEUTRAL_SIGNAL])
        acc = np.float(np.count_nonzero(
            predictions[predictions != NEUTRAL_SIGNAL] == labels[predictions != NEUTRAL_SIGNAL]) / non_neutral)

        return acc, non_neutral

    def computeFeatures(self, bSaveFeatures=False,
                        from_time=None,
                        to_time=None,
                        bComputeIndicators=None,
                        bComputeNormalizedRatios=None,
                        bComputeCandles=None,
                        bComputeHighLowFeatures=None,
                        bBuildXSet=False
                        ):

        self.set_from_to_times(from_time, to_time)

        idx_from = 0
        try:
            while idx_from < len(self.df.index):
                if str(self.df.index[idx_from]) >= from_time:
                    break
                idx_from += 1
            print ('Idx_from: ' + str(idx_from))
        except AttributeError:
            pass
        except:
            LogManager.get_logger ().error("Exception occurred", exc_info=True)

        if idx_from > 600:
            idx_from = idx_from - 600
            from_time = self.df.index[idx_from]
            print ('Idx_from < 600: ' + str(idx_from))

        if self.df is not None:
            f_df = computeFeaturesOnTheFly(self.df.loc[from_time:self.to_time],
                                           timeframe=self.timeframe,
                                           bComputeIndicators=bComputeIndicators,
                                           bComputeNormalizedRatios=bComputeNormalizedRatios,
                                           bComputeCandles=bComputeCandles,
                                           bComputeHighLowFeatures=bComputeHighLowFeatures,
                                           high_low_feat_window = self.high_low_feat_window)
            #if not self.bLoadCandlesOnline:
            #    f_df = f_df.dropna()
            self.f_df = f_df

            if bBuildXSet:
                self.buildXfromFeatDF()
            if bSaveFeatures:
                self.saveFeatures()
                
        return self

    def savePredictions(self, rule_name):
        return self.saveDF2csv(self.p_df, self.predpath,
                        self.pred_filename_prefix + '_' + str(rule_name) + '_',
                        self.pred_filename_suffix)

    def saveCandles(self):
        return self.saveDF2csv(self.df, self.parsedpath,
                        self.parsed_filename_prefix,
                        self.parsed_filename_suffix)

    def saveFeatures(self):
        return self.saveDF2csv(self.f_df, self.featpath,
                        self.feat_filename_prefix,
                        self.feat_filename_suffix)
    

    def saveLabels(self):
        return self.saveDF2csv(self.l_df, self.labelpath,
                        self.label_filename_prefix,
                        self.label_filename_suffix)
        
        
    def saveDF2csv(self, df, path, prefix, suffix='.csv'):
        full_filename = path + '/' + prefix + \
                        self.ccy_pair + '_' + self.timeframe + suffix

        df_complete = df
        # check if file already exists
        try:
            df2 = self.loadCsv2DF(path, prefix, suffix,
                                  self.ccy_pair, self.timeframe, timestamp_filter=False)
            df3 = df2.reset_index().merge(df.reset_index(), how='outer').set_index('Date')
            # df_complete = df3.reset_index().drop_duplicates(subset='Date', keep='last').set_index('Date')

            df_complete = df3[~df3.index.duplicated(keep='last')]
            try:
                del df_complete['index']
            except:
                LogManager.get_logger ().error("Exception occurred", exc_info=True)
            df_complete = indexDf(df_complete)
        except:
            LogManager.get_logger ().error("Exception occurred", exc_info=True)
        df_complete.to_csv(full_filename)
        
        return self

    def loadCandles(self, instrument=None,
                    timeframe=None,
                    bTryToComplete=False,
                    from_time=None, to_time=None):

        self.init_param(instrument, timeframe, from_time, to_time)

        if self.bLoadCandlesOnline:
            self.loadSeriesOnline()
        else:
            try:
                if self.timeframe == 'M15':
                    self.df = self.loadCsv2DF(self.parsedpath,
                                              self.parsed_filename_prefix,
                                              self.parsed_filename_suffix,
                                              self.ccy_pair, self.timeframe)
                else:
                    self.df = self.buildCandlesFromLowerTimeframe()
            except:
                self.df = None
            if bTryToComplete:
                self.completeCandlesDf()
                
        return self

    def completeCandlesDf(self):
        aux_from = self.from_time
        aux_to = self.to_time

        if self.df is None:
            # whole dataset missing
            print ('Candles dataframe was none')
            self.loadSeriesOnline(from_time=self.from_time, to_time=self.to_time)
            self.saveCandles()

        if self.df.shape[0] > 0:
            if str(self.df.index[0]) > self.from_time:
                # completing the beginning
                print ('Completing the beginning of candles dataframe')
                try:
                    self.loadSeriesOnline(from_time=self.from_time, to_time=str(self.df.index[0]))
                    self.saveCandles()
                except:
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)
                self.from_time = aux_from
                self.to_time = aux_to
                self.loadCandles(bTryToComplete=False)
            if str(self.df.index[-1]) < self.to_time:
                # completing the end
                print ('Completing the end of candles dataframe')
                try:
                    self.loadSeriesOnline(from_time=str(self.df.index[-1]),
                                          to_time=self.to_time)
                    self.saveCandles()
                except:
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)
                self.from_time = aux_from
                self.to_time = aux_to
                self.loadCandles(bTryToComplete=False)
        else:
            # whole dataset missing
            print ('Whole candles dataframe was missing')
            self.loadSeriesOnline(from_time=self.from_time, to_time=self.to_time)
            self.saveCandles()

    def loadFeatures(self, instrument=None, timeframe=None, bTryToComplete=False,
                     from_time=None, to_time=None,
                     bBuildXSet=False, bLoadPCA=False):

        self.init_param(instrument, timeframe, from_time, to_time)
        self.f_df = self.loadCsv2DF(self.featpath,
                                    self.feat_filename_prefix,
                                    self.feat_filename_suffix,
                                    self.ccy_pair, self.timeframe)
        
        if bTryToComplete:
            self.completeFeaturesDf()
        if bLoadPCA:
            self.loadPCAFeatures()
        try:
            if bBuildXSet:
                self.buildXfromFeatDF()
        except:
            LogManager.get_logger ().error("Exception occurred", exc_info=True)
            
        return self

    def loadPCAFeatures(self, pca_feat_path=None):
        if pca_feat_path is not None:
            self.pca_feat_path = pca_feat_path
        try:
            pca_df = self.loadCsv2DF(self.pca_feat_path,
                                     '',
                                     '.csv',
                                     self.ccy_pair, self.timeframe)
        except KeyError:
            pca_df = self.loadCsv2DF(self.pca_feat_path,
                                     '',
                                     '.csv',
                                     self.ccy_pair, self.timeframe, index='Unnamed: 0')

        for col in pca_df.columns:
            self.f_df[col.replace(self.ccy_pair + '_' + self.timeframe + '_', '')] = pca_df[col][self.f_df.index]
            
        return self

    def completeFeaturesDf(self):
        aux_from = self.from_time
        aux_to = self.to_time

        if self.f_df is None:
            # whole dataset missing
            print ('Features dataframe was none')
            self.computeFeatures(bSaveFeatures=True,
                                 from_time=self.from_time,
                                 to_time=self.to_time)
            self.saveCandles()

        elif self.f_df.shape[0] > 0:
            if str(self.f_df.index[0]) > self.from_time:
                # completing the beginning
                print ('Completing the beginning of features dataframe')
                print (str(self.f_df.index[0]) + ', ' + self.from_time)
                try:
                    self.computeFeatures(bSaveFeatures=True,
                                         from_time=self.from_time,
                                         to_time=str(self.f_df.index[0]))
                except:
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)
                self.from_time = aux_from
                self.to_time = aux_to
                self.loadFeatures(bTryToComplete=False)
            if str(self.f_df.index[-1]) < self.to_time:
                # completing the end
                print ('Completing the end of features dataframe')
                try:
                    print (str(self.f_df.index[-1]) + ', ' + self.to_time)
                    self.computeFeatures(bSaveFeatures=True,
                                         from_time=str(self.f_df.index[-1]),
                                         to_time=self.to_time)

                except:
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)
                self.from_time = aux_from
                self.to_time = aux_to
                self.loadFeatures(bTryToComplete=False)
        else:
            # whole dataset missing
            print ('Whole features dataframe was missing')
            self.computeFeatures(bSaveFeatures=True,
                                 from_time=self.from_time,
                                 to_time=self.to_time)

    # --------Methods used to build dataset for Machine Learning
    def buildXfromFeatDF(self):
        sent = buildSentencesOnTheFly(self.f_df)
        X = buildSequencePatchesOnTheFly(sent, lookback_window=self.lookback_window)
        self.n_features = X.shape[2]
        self.feat_names_list = self.f_df.columns
        self.X = X
        if self.df is not None:
            self.df = self.df.ix[:, 0:6]

    def buildYfromLabelDF(self):
        labels = self.l_df['Labels']
        self.y = np.zeros((len(labels), 3))
        for i in range(len(labels)):
            self.y[i, 1 - int(NEUTRAL_SIGNAL) + np.int(labels[i])] = 1

    # -------------------------------------------------

    def loadLabels(self, instrument=None, timeframe=None,
                   from_time=None, to_time=None, bBuildY=False):

        self.init_param(instrument, timeframe, from_time, to_time)
        self.l_df = self.loadCsv2DF(self.labelpath,
                                    self.label_filename_prefix,
                                    self.label_filename_suffix,
                                    self.ccy_pair, self.timeframe)

        self.get_active_labels()

        if bBuildY:
            self.buildYfromLabelDF()
        
        return self

    def loadPredictions(self, instrument=None, timeframe=None, rule_name=None,
                        from_time=None, to_time=None):

        self.init_param(instrument, timeframe, from_time, to_time)

        self.p_df = self.loadCsv2DF(self.predpath,
                                    self.pred_filename_prefix + '_' + str(rule_name) + '_',
                                    self.pred_filename_suffix,
                                    self.ccy_pair, self.timeframe)
        
        return self

    def loadCsv2DF(self, path, prefix, suffix,
                   instrument, timeframe, index='Date',
                   from_time=None, to_time=None, timestamp_filter=True):
        self.ccy_pair = instrument
        self.timeframe = timeframe
        full_filename = path + '/' + prefix + \
                        self.ccy_pair + '_' + self.timeframe + suffix
        # print ('Filename: '+ full_filename)
        df = pd.read_csv(full_filename)
        try:
            df = indexDf(df, index)
        except KeyError:
            df = indexDf(df, 'Unnamed: 0')

        if from_time is not None:
            if from_time != '':
                self.from_time = from_time
        if to_time is not None:
            if to_time != '':
                self.to_time = to_time
        if timestamp_filter:
            df = df.loc[self.from_time:self.to_time]

        df = df[~df.index.duplicated(keep='last')]

        return df

    def evaluateRule(self, instrument=None,
                     timeframe=None,
                     rule=Rule('VeryComplex01', rule_veryComplex),
                     from_time=None,
                     to_time=None,
                     bSaveStats=True):
        self.set_from_to_times(from_time, to_time)
        if instrument is not None:
            self.ccy_pair = instrument
        if timeframe is not None:
            self.timeframe = timeframe
        if self.l_df is None:
            self.loadLabels()
        if self.p_df is None:
            try:
                self.loadPredictions(rule_name=rule.name)
            except:
                if rule.ruleType == 'SingleDataset':
                    self.loadFeatures()
                    self.computePredictions(rule=rule)
                else:
                    print (
                        'Predictions not saved and cannot compute predictions for this type of rule, try using DatasetHolder')

        if bSaveStats:
            self.saveRuleStats(rule)

        l = self.l_df['Labels']
        p = self.p_df['Predictions']
        res = 0 * l
        for i in range(len(res)):
            if p[i] != NEUTRAL_SIGNAL:
                if p[i] == l[i]:
                    res[i] = 1 * rule.target_multiple
                else:
                    res[i] = -1

        return res

    def saveRuleStats(self, rule, filename=''):
        n_pred = np.sum((self.p_df.Predictions - NEUTRAL_SIGNAL) ** 2)
        net = np.sum((self.p_df.Predictions - NEUTRAL_SIGNAL) * self.l_df.Labels)

        if filename == '':
            filename = 'Stats_rule_' + rule.name + '.txt'
        f = open(self.predpath + '/' + filename, 'a')
        f.write(self.ccy_pair + ', ' + str(n_pred) + ', ' + str(net) + '\n')
        f.close()

    def createSingleTrainSet(self, y_width=1):
        total_len = 0
        total_len_cv = 0

        lookback_window = self.lookback_window
        n_features = self.n_features

        for i in range(len(self.dataset_list)):
            if (np.size(self.dataset_list[i]) > 0):
                total_len += len(self.dataset_list[i][0])
                total_len_cv += len(self.dataset_list[i][2])

                if n_features != np.size(self.dataset_list[i][0], 2):
                    n_features = np.size(self.dataset_list[i][0], 2)
                if lookback_window != np.size(self.dataset_list[i][0], 1):
                    lookback_window = np.size(self.dataset_list[i][0], 1)

        self.X = np.zeros((total_len, lookback_window, n_features))
        self.y = np.zeros((total_len, y_width))

        self.cv_X = np.zeros((total_len_cv, lookback_window, n_features))
        self.cv_y = np.zeros((total_len_cv, y_width))

        total_len = 0
        total_len_cv = 0
        for i in range(len(self.dataset_list)):
            if (np.size(self.dataset_list[i]) > 0):
                self.X[total_len:total_len + len(self.dataset_list[i][0]), :, :] = self.dataset_list[i][0]
                self.y[total_len:total_len + len(self.dataset_list[i][0]), :] = self.dataset_list[i][1]
                total_len += len(self.dataset_list[i][0])

                self.cv_X[total_len_cv:total_len_cv + len(self.dataset_list[i][2]), :, :] = self.dataset_list[i][2]
                self.cv_y[total_len_cv:total_len_cv + len(self.dataset_list[i][2]), :] = self.dataset_list[i][3]
                total_len_cv += len(self.dataset_list[i][2])

    def loadDataSetV2(self, series_list=None, begin=0, end=10):
        self.dataset_list = []
        self.X = np.zeros((0, self.lookback_window, self.n_features))
        self.y = np.zeros((0, 3))

        if series_list == None:
            series_list = (np.linspace(1, self.total_no_series, self.total_no_series)).astype(int)

        for counter, series_no in enumerate(series_list[begin:end]):
            aux_X = self.X
            aux_y = self.y
            self.loadSeriesByNo(series_no, bLoadOnlyTrainset=True)
            elem = [self.X, self.y, self.cv_X, self.cv_y, self.test_X, self.test_y]
            # self.dataset_list.append (elem)

            # print ('Len aux_X: ' + str(len(aux_X)))

            try:
                self.X = np.zeros((len(elem[0]) + len(aux_X), np.size(aux_X, 1), np.size(aux_X, 2)))
                self.y = np.zeros((len(elem[1]) + len(aux_y), np.size(aux_y, 1)))

                # print ('Len X: ' + str(len(self.X)))

                self.X[0:len(aux_X), :, :] = aux_X
                self.y[0:len(aux_X), :] = aux_y
                self.X[len(aux_X):, :, :] = elem[0]
                self.y[len(aux_X):, :] = elem[1]
            except:
                print ('Error loading series no: ' + str(series_no) + ' to dataset')
                self.X = aux_X
                self.y = aux_y
        return

    def loadDataSet(self, series_list=None, begin=0, end=10, bRelabel=True, bNormalize=True, bLoadTrainset=True,
                    bLoadCvSet=True, bLoadTestSet=True):
        self.dataset_list = []

        if series_list == None:
            series_list = (np.linspace(1, self.total_no_series, self.total_no_series)).astype(int)

        for counter, series_no in enumerate(series_list[begin:end]):
            self.loadSeriesByNo(series_no, bNormalize=bNormalize, bRelabel=bRelabel)
            elem = [self.X, self.y, self.cv_X, self.cv_y, self.test_X, self.test_y]
            self.dataset_list.append(elem)
        return

    def compute_return(self, model, pred_threshold=0.34, ds_sel='cv'):
        if ds_sel == 'cv':
            X = self.cv_X
            y = self.cv_y
        elif ds_sel == 'train':
            X = self.X
            y = self.y
        elif ds_sel == 'test':
            X = self.test_X
            y = self.test_y
        # print ("X shape: "+str(np.shape(X)))
        # print ("y shape: " + str(np.shape(y)))
        return compute_return(model, X, y, threshold=pred_threshold)

    def loadLabelsDataframe(self, series_no, bLoadOnlyTrainset=False):
        # ------------then labels
        label_filename = self.label_filename_prefix + str(
            series_no) + self.label_filename_suffix  # need to fix this name
        my_df = loadSeriesToDataframe(self.labelpath, label_filename)
        # print ("Length labels Dataframe: "+str(len(my_df)))

        # splits dataframe into training set and cross validation sets
        train_labels_df, cv_labels_df, test_labels_df = splitDataframeIntoTrainCVTest(my_df, n_cv=self.cv_set_size,
                                                                                      n_test=self.test_set_size,
                                                                                      bLoadOnlyTrainset=bLoadOnlyTrainset)
        return train_labels_df, cv_labels_df, test_labels_df

    def reloadSeries(self):
        self.loadSeriesByNo()

    def loadSeriesByNo(self, series_no=None, bLoadOnlyTrainset=None, bNormalize=None, bRelabel=None, bConvolveCdl=None):
        self.X = self.y = self.cv_X = self.cv_y = self.test_X = self.test_y = []
        # print ("Loading Series #"+str(series_no))

        if series_no is not None:
            self.series_no = series_no
        elif self.series_no is None:
            print ('Series no cannot be None!')
            return

        if bLoadOnlyTrainset is not None:
            self.bLoadOnlyTrainset = bLoadOnlyTrainset
        if bNormalize is not None:
            self.bNormalize = bNormalize
        if bRelabel is not None:
            self.bRelabel = bRelabel

        # ---------------load series into dataframe
        # parsed_filename = 'ccy_hist_ext_'+str(series_no)+'.txt'
        feat_filename = self.feat_filename_prefix + str(self.series_no) + self.feat_filename_suffix
        label_filename = self.label_filename_prefix + str(
            self.series_no) + self.label_filename_suffix  # need to fix this name

        if True:
            # try:
            self.feat_names_list = loadFeaturesNames(self.featpath, feat_filename)
            # ------------first features
            my_df = loadSeriesToDataframe(self.featpath, feat_filename, self.feat_names)

            # print ("Length Features Dataframe: "+str(len(my_df)))

            # splits dataframe into train, cv and test later
            # train_df, cv_df, test_df = splitDataframeIntoTrainCVTest (my_df, n_cv=self.cv_set_size,
            #                                                          n_test=self.test_set_size,
            #                                                          bLoadOnlyTrainset=bLoadOnlyTrainset)
            train_df = my_df
            if self.bLoadOnlyTrainset == True:
                cv_df = [1]
                test_df = [1]

            # ------------then labels
            if self.bRelabel == False:
                my_df = loadSeriesToDataframe(self.labelpath, label_filename)
                # print ("Length labels Dataframe: "+str(len(my_df)))

                # splits dataframe into training set and cross validation sets
                # train_labels_df, cv_labels_df, test_labels_df = splitDataframeIntoTrainCVTest (my_df, n_cv=self.cv_set_size,
                #                                                          n_test=self.test_set_size,
                #                                                          bLoadOnlyTrainset=bLoadOnlyTrainset)
                train_labels_df = my_df

                print('Train labels shape: ' + str(np.shape(train_labels_df)))

                if self.bLoadOnlyTrainset == True:
                    cv_labels_df = [1]
                    test_labels_df = [1]
            else:
                train_labels_df = None
                cv_labels_df = None
                test_labels_df = None
            # ---------------------------------------------

            # ---------buidSentences does not mean anything, need to get rid of this step
            sentences, next_chars = buildSentences(train_df, train_labels_df,
                                                   bZeroLongDatedMovingAverages=self.bZeroMA_train)
            # if bLoadOnlyTrainset == False:
            #    cv_sentences, cv_next_chars = buildSentences (cv_df, cv_labels_df, bZeroLongDatedMovingAverages=self.bZeroMA_cv)
            #    test_sentences, test_next_chars = buildSentences (test_df, test_labels_df, bZeroLongDatedMovingAverages=self.bZeroMA_test)
            # print ('loadingSeries len cv_sentences:'+str(len(cv_sentences)))
            # ---------initializes train_X, train_y, cv_X and cv_y sets
            self.X, self.y = buildSequencePatches(sentences, next_chars,
                                                  lookback_window=self.lookback_window, no_signals=self.no_signals)
            # if bLoadOnlyTrainset == False:
            #    self.cv_X, self.cv_y = buildSequencePatches (cv_sentences, cv_next_chars,
            #                                                    lookback_window=self.lookback_window, no_signals=self.no_signals)
            #    self.test_X, self.test_y = buildSequencePatches (test_sentences, test_next_chars,
            #                                                    lookback_window=self.lookback_window, no_signals=self.no_signals)

            self.test_X = self.X[-self.test_set_size:, :, :]
            self.test_y = self.y[-self.test_set_size:, :]
            self.cv_X = self.X[-(self.cv_set_size + self.test_set_size):-self.test_set_size, :, :]
            self.cv_y = self.y[-(self.cv_set_size + self.test_set_size):-self.test_set_size, :]
            self.X = self.X[0:-(self.cv_set_size + self.test_set_size), :, :]
            self.y = self.y[0:-(self.cv_set_size + self.test_set_size), :]

            print ("Series #" + str(self.series_no) + "loaded successfully")

            if len(self.X) > self.last:
                self.X = self.X[-self.last:, :, :]
                self.y = self.y[-self.last:, :]

            if self.bRelabel == True:
                if self.period_ahead == 0:
                    period_ahead = np.int(np.random.uniform(low=self.min_period_ahead, high=self.max_period_ahead))
                else:
                    period_ahead = self.period_ahead

                self.y = relabelDataset(self.X, period_ahead=period_ahead, bCenter_y=self.bCenter_y,
                                        return_type=self.ret_type)
                self.cv_y = relabelDataset(self.cv_X, period_ahead=period_ahead, bCenter_y=self.bCenter_cv_y,
                                           return_type=self.ret_type)
                self.test_y = relabelDataset(self.test_X, period_ahead=period_ahead, bCenter_y=self.bCenter_test_y,
                                             return_type=self.ret_type)

            if self.bNormalize == True:

                try:
                    self.X = normalizeOnTheFly(self.X, mu_sigma_list=self.mu_sigma_list, by100_list=self.by100_list,
                                               volume_feat_list=self.volume_feat_list)

                    self.cv_X = normalizeOnTheFly(self.cv_X, mu_sigma_list=self.mu_sigma_list,
                                                  by100_list=self.by100_list, volume_feat_list=self.volume_feat_list)

                    self.test_X = normalizeOnTheFly(self.test_X, mu_sigma_list=self.mu_sigma_list,
                                                    by100_list=self.by100_list, volume_feat_list=self.volume_feat_list)

                except:
                    print ('Error normalizing Dataset')

            if self.bConvolveCdl == True:
                self.convolveCdl()

        # except:
        # print ("Failed to load Series #"+str(series_no))

    def convolveCdl(self):
        for cdl in get_TA_CdL_Func_List ():
            i = self.getFeatIdx(cdl)
            exp_fn = np.exp(-np.linspace(0, 40, 20) / 4)

            self.X[:, -1, i] = np.convolve(self.X[:, -1, i], exp_fn)[:len(self.X)]
            self.cv_X[:, -1, i] = np.convolve(self.cv_X[:, -1, i], exp_fn)[:len(self.cv_X)]
            self.test_X[:, -1, i] = np.convolve(self.test_X[:, -1, i], exp_fn)[:len(self.test_X)]

    def dropCandleStickFeatures(self, cdl_min_idx=81):
        idx_list = []
        comp_list = []

        for cdl in get_TA_CdL_Func_List ():
            idx = self.getFeatIdx(cdl)
            idx_list.append(idx)

        for i in range(np.shape(self.X)[2]):
            if i not in idx_list:
                comp_list.append(i)

        self.X = self.X[:, :, comp_list]
        self.cv_X = self.cv_X[:, :, comp_list]
        self.test_X = self.test_X[:, :, comp_list]

    def evaluateOnLoadedDataset(self, model, show_plots=True):
        my_shape = (len(self.dataset_list))

        self.ret_array = np.zeros(my_shape)
        self.long_hits_array = np.zeros(my_shape)
        self.long_misses_array = np.zeros(my_shape)
        self.short_hits_array = np.zeros(my_shape)
        self.short_misses_array = np.zeros(my_shape)
        self.loss_array = np.zeros(my_shape)
        self.val_loss_array = np.zeros(my_shape)

        for k in range(len(self.dataset_list)):
            if np.size(self.dataset_list[k]) == 6:
                [self.X, self.y, self.cv_X, self.cv_y, self.test_X, self.test_y] = self.dataset_list[k]
                # self.ret_array[k], self.long_hits_array[k], self.long_misses_array[k], self.short_hits_array[k], self.short_misses_array[k] = compute_return(self.model,self.cv_X[:-np.mod(len(self.cv_X),batch_size),:,:], self.cv_y[:-np.mod(len(self.cv_X),batch_size)])
                self.ret_array[k], self.long_hits_array[k], self.long_misses_array[k], self.short_hits_array[k], \
                self.short_misses_array[k] = compute_return(model, self.cv_X, self.cv_y)
        if show_plots == True:
            fig = plt.figure()
            # for k in range(len (self.dataset_list)):
            plt.plot(self.ret_array, label="Returns")
            plt.legend(loc='best')
            plt.show()

    def getFeatIdx(self, feat='Close', func=None, args=None, verbose=False):
        if verbose:
            print ('Looking for ' + str(feat))
        self.feat_names_list = list(self.feat_names_list)

        if feat in self.feat_names_list:
            if verbose:
                print ('Found')
            return self.feat_names_list.index(feat)
        elif func is not None:
            # new_feat = np.zeros ((np.shape(self.X)[0], np.shape(self.X)[1], 1))
            new_feat = func(self, args=args)

            self.X = np.concatenate((self.X, new_feat), axis=2)
            self.f_df[feat] = np.reshape(new_feat, len(new_feat))
            self.feat_names_list.append(feat)
            return self.feat_names_list.index(feat)
        else:
            if verbose:
                print ('Not found')
            return 9999

    # method to download oanda data for all instruments for a given timeframe
    def download_oanda_data(self, since=2000):
        self.initOnlineConfig()

        from_time = dt.datetime(since, 1, 1, 0, 0, 0)

        while from_time < dt.datetime.today():
            to_time = from_time + relativedelta(months=1)

            self.set_from_to_times(from_time=str(from_time),
                                   to_time=str(to_time))
            try:
                self.loadSeriesOnline(bSaveCandles=True)
                print ('Loaded data succesfully for ' + str(from_time))
            except:
                print ('Failed to load data for ' + str(from_time))
            from_time = to_time

        # need to check if all dates were loaded
        self.set_from_to_times(from_time=2012, to_time=2018)
        self.loadCandles()
        full_dt_list = []
        from_time = dt.datetime(2012, 1, 1, 0, 0, 0)
        while from_time < dt.datetime.today():
            if from_time.weekday() < 5:
                full_dt_list.append(from_time)
            else:
                print ('Trying to load missing data for ' + str(from_time))
                try:
                    self.loadSeriesOnline(from_time=str(from_time + relativedelta(days=-9)),
                                          to_time=str(from_time + relativedelta(days=9)),
                                          bSaveCandles=True)
                    print ('len(df): ' + str(len(self.df)))
                    from_time += relativedelta(days=7)
                except:
                    print ('Failed to load data for ' + str(from_time))
            from_time += relativedelta(days=1)

        self.loadCandles(from_time=2000, to_time=str(dt.datetime.today()))
        dts_loaded = list(set([str(dt_str)[0:10] for dt_str in self.df.index]))
        dts_checked = [str(my_date)[0:10] in dts_loaded for my_date in full_dt_list]
        plt.plot(full_dt_list, dts_checked)

        self.computeFeatures(bSaveFeatures=True)
        
        return self