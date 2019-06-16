# -*- coding: utf-8 -*-

from Framework.Dataset.DatasetHolder import *
from Config.const_and_paths import *

import pandas as pd

def plot_signals (ds, bMultiple = False, bSave=False, label = '', plot_filename=''):
    if not bMultiple:
        fig = plt.figure ()
    plt.title ('Signals')
    axes = plt.gca()
    ax = axes.twinx ()
    axes.plot(ds.p_df.Predictions, color='red')
    ax.plot(ds.f_df['Close_'+ds.timeframe])
    
    if not bMultiple:
        plt.show ()
        if bSave:
            fig.savefig(plot_filename)    

def plot_pnl (ds, bMultiple = False, bSave=False, label = '', plot_filename=''):
    if not bMultiple:
        fig = plt.figure ()
    plt.title ('PnL')
    #pnl = np.array([1 if label == pred else -1 for label, pred in zip (ds.l_df.Labels, ds.p_df.Predictions)])
    #plt.plot (np.cumsum(pnl))
    plt.plot(np.cumsum(ds.l_df.Labels * ds.p_df.Predictions), label = label)
    if not bMultiple:
        plt.show ()
        if bSave:
            fig.savefig(plot_filename)    
    
def plot_histogram (ds, bSave=False, plot_filename=''):
    #fig = plt.figure ()
    plt.title ('Hit ratio - Histogram')
    plt.hist(ds.l_df.Labels[ds.p_df.Predictions != 0] * ds.p_df.Predictions[ds.p_df.Predictions != 0], bins=5)
    #plt.show ()
    if bSave:
        fig.savefig(plot_filename)

def compute_predictions_simple (ds):
    df = ds.f_df #just a short name
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    #df['trendlines_diff_10_D'] = df['no_standing_upward_lines_10_D'] - df['no_standing_downward_lines_10_D']
    #features below look at the difference between current value of the above feature and its max/min
    #df['trend_diff_change_up_D'] = (df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).min())
    #df['trend_diff_change_down_D'] = -(df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).max())

    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI_' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).min ()
    df['max_RSI' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))

    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    preds[(df['RSI_' + fast_timeframe] < 30) & \
          (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe ] > 50)] = 1
   
    #sells if:
    #opposite of above
    preds[(df['RSI_' + fast_timeframe] > 70) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50)] = -1    
    

    return preds


#This is the core of the strategy, 
#the bit that generates signals based on the features up to time t

def compute_predictions (ds, **kwargs):
    #ds = kwargs['ds']
    
    if 'fast_timeframe' in kwargs.keys ():
        fast_timeframe = kwargs['fast_timeframe']
    else:
        fast_timeframe = 'M15'
        
    if 'slow_timeframe' in kwargs.keys ():
        slow_timeframe = kwargs['slow_timeframe']
    else:
        slow_timeframe = 'D'
    
    df = ds.f_df #just a short name
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    df['trendlines_diff_10_D'] = df['no_standing_upward_lines_10_D'] - df['no_standing_downward_lines_10_D']
    #features below look at the difference between current value of the above feature and its max/min
    df['trend_diff_change_up_D'] = (df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).min())
    df['trend_diff_change_down_D'] = -(df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).max())
    
    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI_' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).min ()
    df['max_RSI' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))
    
    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    preds[(df['RSI_' + fast_timeframe] < 30) & \
          (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe ] > 50) & \
          (df['trendlines_diff_10_D'] > 5) & \
          (df['trend_diff_change_down_D'] <= 3)] = 1
    preds[(df['RSI_' + fast_timeframe] < 30) & \
          (df['RSI_' + fast_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe] > 50) & \
          (df['trend_diff_change_up_D'] >= 5) & \
          (df['trendlines_diff_10_D'] > -5)] = 1.0
    
    #sells if:
    #opposite of above
    preds[(df['RSI_' + fast_timeframe] > 70) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50) & (df['trendlines_diff_10_D'] < -5) & (df['trend_diff_change_up_D'] <= 3)] = -1    
    preds[(df['RSI_' + fast_timeframe] > 70) & (df['RSI_' + fast_timeframe] > 30) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50) & (df['trend_diff_change_down_D'] >= 5) & (df['trendlines_diff_10_D'] < 5)] = -1.0

    return preds
    
def compute_predictions_v2 (ds):
    df = ds.f_df #just a short name
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    df['trendlines_diff_10_D'] = df['no_standing_upward_lines_10_D'] - df['no_standing_downward_lines_10_D']
    #features below look at the difference between current value of the above feature and its max/min
    df['trend_diff_change_up_D'] = (df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).min())
    df['trend_diff_change_down_D'] = -(df.trendlines_diff_10_D - df.trendlines_diff_10_D.rolling(window=2000).max())

    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI'] = df.RSI.rolling(window=10).min ()
    df['max_RSI'] = df.RSI.rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))

    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    preds[(df['RSI'] < 40) & (df['RSI_' + slow_timeframe] < 70) & \
          (df['RSI_' + slow_timeframe ] > 30) & (df['trendlines_diff_10_D'] > 5) & (df['trend_diff_change_down_D'] <= 5)] = 1
    preds[(df['RSI'] < 40) & (df['RSI'] < 70) & (df['RSI_' + slow_timeframe] < 70) & (df['RSI_' + slow_timeframe] > 30) & (df['trend_diff_change_up_D'] >= 5) & (df['trendlines_diff_10_D'] > -5)] = 1.0

    #sells if:
    #opposite of above
    preds[(df['RSI'] > 60) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 70) & (df['trendlines_diff_10_D'] < -5) & (df['trend_diff_change_up_D'] <= 5)] = -1    
    preds[(df['RSI'] > 60) & (df['RSI'] > 30) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 70) & (df['trend_diff_change_down_D'] >= 5) & (df['trendlines_diff_10_D'] < 5)] = -1.0

    return preds

def compute_predictions_pca (ds, **kwargs):
    df = ds.f_df #just a short name
        
    if 'fast_timeframe' in kwargs.keys ():
        fast_timeframe = kwargs['fast_timeframe']
    else:
        fast_timeframe = 'M15'
        
    if 'slow_timeframe' in kwargs.keys ():
        slow_timeframe = kwargs['slow_timeframe']
    else:
        slow_timeframe = 'D'
    
    #adding a new feature
    #computes the difference between the number of upward trendlines and downward lines which have not yet been broken
    #positive means more upward lines, possibly uptrend
    #negative means the opposite
    df['trendlines_diff_10_' + slow_timeframe] = df['no_standing_upward_lines_10_' + slow_timeframe] - df['no_standing_downward_lines_10_' + slow_timeframe]
    #features below look at the difference between current value of the above feature and its max/min
    df['trend_diff_change_up_' + slow_timeframe] = (df['trendlines_diff_10_' + slow_timeframe] - df['trendlines_diff_10_' + slow_timeframe].rolling(window=2000).min())
    df['trend_diff_change_down_' + slow_timeframe] = - (df['trendlines_diff_10_' + slow_timeframe] - df['trendlines_diff_10_' + slow_timeframe ].rolling(window=2000).max())
    
    #min/max rolling RSI over a window of 10 periods of the fast timeframe
    #useful to check if RSI off the lows
    df['min_RSI_' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).min ()
    df['max_RSI' + fast_timeframe] = df['RSI_' + fast_timeframe].rolling(window=10).max ()

    #computing predictions
    preds = np.zeros(len(ds.f_df))
    
    #buys if:
    #RSI < 40 (could be lower threshold) => fading weakness
    #RSI(slow timeframe) < 70 => not overbought
    #RSI(slow timeframe) > 50 => strength on the lower frequency timeframe
    #trendline_diff_D > X => trending up
    #trend_diff_change_down_D < Y => means that recently, upward lines have not been broken nor new downward lines have been forged
    criteria = np.ones (len(ds.f_df))
    
    #minimum correlation to the first principal component
    if 'rho_min' in kwargs.keys (): 
        rho_min = kwargs['rho_min']
    else:
        rho_min = 0.4
        
    #residual of the regression versus the first principal component
    if 'resid_threshold' in kwargs.keys (): 
        resid_threshold = kwargs['resid_threshold']
    else:
        resid_threshold = 5.0
    
    #maximum RSI fast value to allow to trigger long position
    if 'RSI_fast_max' in kwargs.keys (): 
        RSI_fast_max = kwargs['RSI_fast_max']
    else:
        RSI_fast_max = 50.0
        
    #minimum RSI fast value to allow to trigger long position
    if 'RSI_fast_min' in kwargs.keys (): 
        RSI_fast_min = kwargs['RSI_fast_min']
    else:
        RSI_fast_min = 30.0
        
    #maximum RSI slow value to allow to trigger long position
    if 'RSI_slow_max' in kwargs.keys (): 
        RSI_slow_max = kwargs['RSI_slow_max']
    else:
        RSI_slow_max = 70.0
        
    #minimum RSI slow value to allow to trigger long position
    if 'RSI_slow_min' in kwargs.keys (): 
        RSI_slow_min = kwargs['RSI_slow_min']
    else:
        RSI_slow_min = 50.0
        
    #net trendlines min: minimum difference between the number of upward lines and downward lines to trigger a long position
    if 'net_trendlines_min' in kwargs.keys (): 
        net_trendlines_min = kwargs['net_trendlines_min']
    else:
        net_trendlines_min = 5.0
        
    #maximum number of upward trendlines broken or new downward lines to trigger long position
    if 'trendlines_delta' in kwargs.keys (): 
        trendlines_delta = kwargs['trendlines_delta']
    else:
        trendlines_delta = 5.0
        
    if 'criterium' in kwargs.keys ():
        criterium = kwargs['criterium']
    else:
        criterium = 'both'
    
    if criterium == 'first' or criterium == 'both':
        preds[ (np.abs(df['rho_' + fast_timeframe]) > rho_min) & \
              (df['n_resid_' + fast_timeframe] < - resid_threshold) & \
              (df['RSI_' + fast_timeframe] < RSI_fast_max) & \
              (df['RSI_' + fast_timeframe] > RSI_fast_min) & \
              (df['RSI_' + slow_timeframe] < RSI_slow_max) & \
              (df['RSI_' + slow_timeframe ] > RSI_slow_min) & \
              (df['trendlines_diff_10_' + slow_timeframe] >= net_trendlines_min) & \
              (df['trend_diff_change_down_' + slow_timeframe] <= trendlines_delta)] = 1
    
    if criterium == 'second' or criterium == 'both':      
        preds[(np.abs(df['rho_' + fast_timeframe]) > rho_min) & \
              (df['n_resid_' + fast_timeframe] < -resid_threshold) & \
              (df['RSI_' + fast_timeframe] < RSI_fast_max) & \
              (df['RSI_' + fast_timeframe] > RSI_fast_min) & \
              (df['RSI_' + slow_timeframe] < RSI_slow_max) & \
              (df['RSI_' + slow_timeframe] > RSI_slow_min) & \
              (df['trend_diff_change_up_' + slow_timeframe] >= trendlines_delta) & \
              (df['trendlines_diff_10_' + slow_timeframe] > -net_trendlines_min)] = 1.0
        
    #sells if:
    #opposite of above
    if criterium == 'first' or criterium == 'both':
        preds[(np.abs(df['rho_' + fast_timeframe]) > rho_min) & \
              (df['n_resid_' + fast_timeframe] > resid_threshold) & \
              (df['RSI_' + fast_timeframe] > (100.0 - RSI_fast_max)) & \
              (df['RSI_' + fast_timeframe] < (100.0 - RSI_fast_min)) & \
              (df['RSI_' + slow_timeframe] > (100.0 - RSI_slow_max)) & \
              (df['RSI_' + slow_timeframe] < (100.0 - RSI_slow_min)) & \
              (df['trendlines_diff_10_' + slow_timeframe] <= -net_trendlines_min) & \
              (df['trend_diff_change_up_' + slow_timeframe] <= trendlines_delta)] = -1   
              
    if criterium == 'second' or criterium == 'both':      
        preds[(np.abs(df['rho_' + fast_timeframe]) > rho_min) & \
              (df['n_resid_' + fast_timeframe] > resid_threshold) & \
              (df['RSI_' + fast_timeframe] > (100.0 - RSI_fast_max) ) & \
              (df['RSI_' + fast_timeframe] < (100.0 - RSI_fast_min)) & \
              (df['RSI_' + slow_timeframe] > (100.0 - RSI_slow_max)) & \
              (df['RSI_' + slow_timeframe] < (100.0 - RSI_slow_min)) & \
              (df['trend_diff_change_down_' + slow_timeframe] >= trendlines_delta) & \
              (df['trendlines_diff_10_' + slow_timeframe] > net_trendlines_min)] = -1.0

    return preds

class VectorizedStrategy ():
    def __init__ (self, timeframe='M15', 
                  other_timeframes=['D'],
                  from_time=2006, 
                  to_time=2015,
                  ds=None):
        
        self.from_time = None
        self.to_time = None
        set_from_to_times (self, from_time, to_time)
        
        self.timeframe = timeframe #trading timeframe
        self.other_timeframes = other_timeframes
        self.pnl_df = None
        #init pnl dataframe
        ds = Dataset(ccy_pair='EUR_USD', 
                     from_time=self.from_time, 
                     to_time=self.to_time, timeframe=timeframe)
        ds.loadCandles()
        self.pnl_df = pd.core.frame.DataFrame(index=ds.df.index)
    
    def load_instrument (self, instrument='USD_ZAR'):
        self.dsh = DatasetHolder(from_time =self.from_time, 
                                 to_time=self.to_time, 
                                 instrument=instrument)
        
        timeframe_list = self.other_timeframes + [self.timeframe]
        
        self.dsh.loadMultiFrame(ccy_pair_list = [instrument], 
                                timeframe_list = timeframe_list, 
                                bComputeFeatures=[tf != 'D' for tf in timeframe_list], 
                                bLoadFeatures=[tf == 'D' for tf in timeframe_list])
        try:
            self.dsh.ds_dict[instrument+'_'+self.timeframe].loadPCAFeatures ()
        except:
            pass
        
        self.ds = self.dsh.ds_dict[instrument+'_M15']

        self.dsh.appendTimeframesIntoOneDataset (lower_timeframe=self.timeframe, 
                                                 daily_delay=0)
        
    def compute_predictions (self, func=compute_predictions_pca, **kwargs):
        preds = func(self.ds, **kwargs)
        self.ds.set_predictions(preds)
        
        return self
        
    def plot_pnl (self, bMultiple = False, bSave=False, label = '', plot_filename=''):
        plot_pnl (self.ds, bMultiple = bMultiple, bSave=bSave,
                  label = label, plot_filename = plot_filename)
        
        return self
        
    def plot_signals (self, plot_filename=''):
        plot_signals (self.ds, plot_filename=plot_filename)
        
        return self
        
    def plot_hist (self, plot_filename=''):
        plot_histogram (self.ds, plot_filename=plot_filename)

        return self

plots_path = u'./Analysis/Results/Strategies/Vectorized/Trendlines_and_change_RSI_2_timeframes'
ccy = 'AUD_USD'
slow_timeframe = 'D'
fast_timeframe_list = ['M15']
daily_delay = 5     #to avoid look-ahead bias
serial_gap_list = [0] #, 20, 80, 160]   #to remove serial predictions
from_time = 2004
to_time = 2014

if False:
    ds = Dataset(ccy_pair='EUR_USD', from_time=2006, to_time=2015)
    ds.loadCandles()
    
    fast_timeframe = 'M15'
    pnl = pd.core.frame.DataFrame(index=ds.df.index)
    for ccy in full_instrument_list:
        try:
            dsh = DatasetHolder(from_time =2006, to_time=2015, instrument=ccy)
            dsh.loadMultiFrame(ccy_pair_list=[ccy], timeframe_list=['D', 'M15'], bComputeFeatures=[False, True], bLoadFeatures=[True, False])
            dsh.ds_dict[ccy+'_M15'].loadPCAFeatures ()
            ds_d = dsh.ds_dict[ccy+'_D']
            ds = dsh.ds_dict[ccy+'_M15']
    
            dsh.appendTimeframesIntoOneDataset (lower_timeframe='M15', daily_delay=0)
            
            preds = compute_predictions(ds)
            ds.set_predictions(preds)
            #ds.removeSerialPredictions()
            plot_pnl(ds)
            pnl[ccy] = np.cumsum(ds.l_df.Labels * ds.p_df.Predictions)[pnl.index]
        except:
            pass
    plt.plot(pnl.sum(axis=1))

if False:
    for fast_timeframe in fast_timeframe_list:

        for ccy in full_instrument_list:
            try:
                dsh = DatasetHolder(from_time=from_time, 
                                    to_time=to_time, 
                                    instrument=ccy)
                dsh.loadMultiFrame ()
                dsh.appendTimeframesIntoOneDataset(lower_timeframe=fast_timeframe,
                                                   higher_timeframe=slow_timeframe)
                
                ds_f = dsh.ds_dict[ccy+'_'+fast_timeframe]
                preds = compute_predictions (ds_f)
                
                for serial_gap in serial_gap_list:
                    ds_f.set_predictions(preds) #uses deepcopy, creates ds.p_df
                    if serial_gap != 0:
                        ds_f.removeSerialPredictions(serial_gap)
                
                    plot_signals (ds_f, True, plots_path + '/Signals_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_pnl (ds_f, True, plots_path + '/PnL_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_histogram (ds_f, True, plots_path + '/Histogram_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                
                
            except:
                pass
            
if False:
    for fast_timeframe in fast_timeframe_list:
        for ccy in fx_list:

            try:
                ds = Dataset (ccy_pair = ccy, 
                              timeframe = fast_timeframe,
                              from_time = from_time,
                              to_time = to_time)
                ds.loadFeatures ()
                preds = compute_predictions (ds)
            except:
                try:                
                    dsh = DatasetHolder(from_time=from_time, 
                                    to_time=to_time, instrument=ccy)
                    try:
                        dsh.loadMultiFrame(timeframe_list=[slow_timeframe, fast_timeframe])
                        ds = dsh.ds_dict[ccy + '_' + fast_timeframe]
                        ds.computeFeatures (bComputeIndicators=True,
                                                 bComputeNormalizedRatios=True,
                                                 bComputeCandles=False,
                                                 bComputeHighLowFeatures=False)
                    except:                        
                        ds = Dataset (ccy_pair = ccy, 
                              timeframe = fast_timeframe,
                              from_time = from_time,
                              to_time = to_time)
                        ds.computeFeatures (bComputeIndicators=True,
                                                 bComputeNormalizedRatios=True,
                                                 bComputeCandles=False,
                                                 bComputeHighLowFeatures=False)
                        ds.saveFeatures ()
                        try:
                            ds.loadLabels ()
                            assert (len(ds.f_df) == len (ds.l_df))
                        except:
                            ds.computeLabels ()
                            sa.saveLabels ()
                        dsh.loadMultiFrame(timeframe_list=[slow_timeframe, fast_timeframe])
                        
                    
                    
                    
                    #appends the slow timeframe columns to the fast timeframe one
                    dsh.appendTimeframesIntoOneDataset(instrument = ccy, 
                                                       lower_timeframe = fast_timeframe,
                                                       daily_delay=daily_delay)
                    ds = dsh.ds_dict[ccy + '_' + fast_timeframe]
                    ds.saveFeatures ()
                    preds = compute_predictions (ds)
                except:
                    pass
                    
            try:
                ds.loadLabels ()
                labels = ds.get_active_labels ()
                
                assert (len(ds.f_df) == len (ds.l_df))

            except:
                ds.computeLabels ()
                ds.saveLabels ()
                
            try:    
                for serial_gap in serial_gap_list:
                    ds.set_predictions(preds) #uses deepcopy, creates ds.p_df
                    if serial_gap != 0:
                        ds.removeSerialPredictions(serial_gap)
                
                    plot_signals (ds, True, plots_path + '/Signals_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_pnl (ds, True, plots_path + '/PnL_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
                    
                    plot_histogram (ds, True, plots_path + '/Histogram_' + 
                                ccy + '_' + slow_timeframe + '_' + 
                                fast_timeframe + 
                                ('' if serial_gap == 0 else '_' + 
                                 str(serial_gap)) + '.png')
            except:
                print 'An error ocurred: ' + ccy + fast_timeframe