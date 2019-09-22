# -*- coding: utf-8 -*-

#this simulates the full strategy, not only entry points
#takes into account stop and target functions

from Framework.Dataset.Dataset import *
from Framework.Genetic.Chromossome import *
from View import *
import numpy as np

def fn_stop_update_trailing_v1 (trade):
    idx = trade.indices[-1]
    
    trailing_bars = parse_kwargs (['trailing_bars'], 5, **trade.kwargs)
    move_proportion = parse_kwargs (['move_proportion'], 0.5, **trade.kwargs)
    
    if trade.direction == LONG_SIGNAL:
        trailing_low = np.min (trade.ds.p_df.Low[idx-2:idx+1])
        if trailing_low > trade.stops[-1]:
            return (trade.stops[-1] + 0.5 * (trailing_low - trade.stops[-1]))
        else:
            return trade.stops[-1]
    else:
        trailing_high = np.max (trade.ds.p_df.High[idx-2:idx+1])
        if trailing_high < trade.stops[-1]:
            return (trade.stops[-1] + 0.5 * (trailing_high - trade.stops[-1]))
        else:
            return trade.stops[-1]
    return

class TradeSimulation ():
    def __init__ (self, ds,
                  direction, 
                  start_idx,
                  **kwargs):
        
        self.ds = ds
        self.direction = direction  
        self.start_idx = start_idx
        self.start_t = parse_kwargs (['start_t'], None, **kwargs)
        self.quantity = parse_kwargs (['quantity'], 1, **kwargs)
        self.func_init_stop = func_init_stop = parse_kwargs (['func_init_stop'], None, **kwargs)
        self.func_init_target = func_init_target = parse_kwargs (['func_init_target'], None, **kwargs)
        self.func_update_stop = func_update_stop = parse_kwargs (['func_update_stop'], None, **kwargs)
        self.func_update_target = func_update_target = parse_kwargs (['func_update_target'], None, **kwargs)
        self.kwargs = kwargs
        
        #ToDo: need to check if stop not hit before trade opening
        if self.start_t is None:
            self.start_t = self.ds.p_df.index[self.start_idx]
        self.start_px = self.ds.p_df.Open[self.start_idx]
        
        self.open_lag = []
        self.high_lag = []
        self.low_lag = []
        self.close_lag = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
    
        self.ts = [self.start_t]
        self.indices = [self.start_idx]
        if func_init_stop is None:
            self.stops = ([self.ds.p_df.Low[self.start_idx - 1]] if self.direction == LONG_SIGNAL else [self.ds.p_df.High[self.start_idx - 1]])
        else:
            self.stops = [func_init_stop (self)] 
        
        if func_init_target is None:
            self.targets = ([self.start_px * 2.00] if self.direction == LONG_SIGNAL else [self.start_px * 0.98])
        else:
            self.targets = [func_init_target (self)]
                
        self.func_update_stop = func_update_stop
        self.func_update_target = func_update_target        
        self.isAlive = True
        self.isValid = True #checks whether trade not stopped before start
        self.mtms = []
        self.events = []
        
    def __str__ (self):
        outstr = 't: ' + str (self.ts) + '\n'
        outstr += 'event: ' + str (self.events) + '\n'
        outstr += 'mtm: ' + str (self.mtms) + '\n'
        outstr += 'stop: ' + str (self.stops) + '\n'
        outstr += 'target: ' + str (self.targets) + '\n'
        
        outstr += 'open_lag: ' + str (self.open_lag) + '\n'
        outstr += 'high_lag: ' + str (self.high_lag) + '\n'
        outstr += 'low_lag: ' + str (self.low_lag) + '\n'
        outstr += 'close_lag: ' + str (self.close_lag) + '\n'
        
        outstr += 'open: ' + str (self.open) + '\n'
        outstr += 'high: ' + str (self.high) + '\n'
        outstr += 'low: ' + str (self.low) + '\n'
        outstr += 'close: ' + str (self.close) + '\n'
        
        return outstr
    
    def plot (self, bTarget = True):
        o = self.ds.p_df.Open[self.start_idx - 10:self.start_idx + len (self.stops) + 10]
        h = self.ds.p_df.High[self.start_idx - 10:self.start_idx + len (self.stops) + 10]
        l = self.ds.p_df.Low[self.start_idx - 10:self.start_idx + len (self.stops) + 10]
        c = self.ds.p_df.Close[self.start_idx - 10:self.start_idx + len (self.stops) + 10]
        
        fig = plot_candles (o=o, h=h, l=l, c=c, bShow = False)
        plt.plot (range (16, 16 + len (self.stops) + 10), self.start_px * np.ones (len (self.stops) + 10), color='black', linestyle='--', label = 'entry')
        plt.plot (range (16, 16 + len (self.stops)), self.stops, color='red', label = 'stop')
        if bTarget:
            plt.plot (range (16, 16 + len (self.targets)), self.targets, color='green', label = 'targets')
        #plt.grid (True)
        plt.legend (loc='best')
        plt.show ()
        
    def run_trade_by_one_bar (self, bVerbose = False):
        if not self.isAlive:
            return
        
        #checks if trade not stopped before start
        if (self.start_px <= self.stops[0] and self.direction == LONG_SIGNAL) or \
            (self.start_px >= self.stops[0] and self.direction == SHORT_SIGNAL):
            self.isAlive = False
            self.isValid = False
            self.mtms = [0.0]
            self.events += ['Stopped before start']
            return
        
        #checks if trade did not hit target before start
        if (self.start_px >= self.targets[0] and self.direction == LONG_SIGNAL) or \
            (self.start_px <= self.targets[0] and self.direction == SHORT_SIGNAL):
            self.isAlive = False
            self.isValid = False
            self.mtms = [0.0]
            self.events += ['Hit target before start']
            return
        
        self.open_lag += [self.ds.p_df.Open[self.indices[-1] - 1]]
        self.high_lag += [self.ds.p_df.High[self.indices[-1] - 1]]
        self.low_lag += [self.ds.p_df.Low[self.indices[-1] - 1]]
        self.close_lag += [self.ds.p_df.Close[self.indices[-1] - 1]]
        
        self.open += [self.ds.p_df.Open[self.indices[-1]]]
        self.high += [self.ds.p_df.High[self.indices[-1]]]
        self.low += [self.ds.p_df.Low[self.indices[-1]]]
        self.close += [self.ds.p_df.Close[self.indices[-1]]]
        
        #checks whether stop hit
        #then, checks whether target hit
        if self.direction == LONG_SIGNAL:
            if self.stops [-1] >= self.ds.p_df.Low[self.ts[-1]]:
                #trade hit stop
                self.events.append ('Stop hit')
                self.isAlive = False
                self.mtms.append ((self.stops[-1] / self.start_px - 1) * self.quantity)
                return
            if self.targets [-1] <= self.ds.p_df.High[self.ts[-1]]:
                #trade hit stop
                self.events.append ('Target hit')
                self.isAlive = False
                self.mtms.append ((self.targets[-1] / self.start_px - 1) * self.quantity)
                return
        else:
            if self.stops [-1] <= self.ds.p_df.High[self.ts[-1]]:
                #trade hit stop
                self.events.append ('Stop hit')
                self.isAlive = False
                self.mtms.append (-(self.stops[-1] / self.start_px - 1) * self.quantity)
                return
            if self.targets [-1] >= self.ds.p_df.Low[self.ts[-1]]:
                #trade hit stop
                self.events.append ('Target hit')
                self.isAlive = False
                self.mtms.append (-(self.targets[-1] / self.start_px - 1) * self.quantity)
                return
                
        #trade survived
        self.events.append ('Survived')
        self.mtms.append ((self.ds.p_df.Close[self.ts[-1]] / self.start_px - 1) * self.quantity * (1 if self.direction == LONG_SIGNAL else -1))
        self.indices.append (self.indices[-1] + 1)
        try:
            self.ts.append (self.ds.p_df.index[self.indices[-1]])
        except:
            self.events [-1] = 'Hit right edge'
            self.isAlive = False
            return
        
        if self.func_update_stop is not None:
            self.stops.append (self.func_update_stop (self))
        else:
            self.stops.append ( self.stops [-1] )
            
        if self.func_update_target is not None:
            self.targets.append (self.func_update_target (self))
        else:
            self.targets.append ( self.targets [-1] )
        
        if bVerbose:
            print (self)

if False:
    MIN_BAR_LENGTH = 0.001
    MIN_CANDLE_BODY_RATIO = 2.5
    ds = Dataset(ccy_pair = 'SPX500_USD',
                 timeframe='M15', from_time = 2000, to_time = 2015)
    
    ds.loadCandles ()
    ds.computeFeatures(bComputeCandles=True)
    
    signals = ((ds.f_df.CDLHAMMER == 1) & \
               (np.abs((ds.f_df.High - ds.f_df.Low) / (ds.f_df.Open - ds.f_df.Close) >= MIN_CANDLE_BODY_RATIO)) & \
                        (((ds.f_df.High - ds.f_df.Low) / ds.f_df.Open) >= MIN_BAR_LENGTH)) * LONG_SIGNAL
                
    signals += ((ds.f_df.CDLINVERTEDHAMMER == 1) & \
                (np.abs((ds.f_df.High - ds.f_df.Low) / (ds.f_df.Open - ds.f_df.Close) >= MIN_CANDLE_BODY_RATIO)) & \
                        (((ds.f_df.High - ds.f_df.Low) / ds.f_df.Open) >= MIN_BAR_LENGTH)) * SHORT_SIGNAL
    
    ds.set_predictions(signals)
    #ds.removeSerialPredictions (60)
    
if True:
    ds = Dataset(ccy_pair='SPX500_USD', 
                              from_time = 2000,
                              to_time=2013, 
                              timeframe='M15')
    ds.loadCandles ()
    ds.computeFeatures ()
    #creates a chromossome
    c = Chromossome (ds = ds, bDebug = True, bSlim = False)

    
    c.add_gene(timeframe = 'M15',
                     func_dict = {'hammer':{'func':fn_hammer, 
                                          'kwargs':{'conv_window': 10,
                                                    'MIN_BAR_LENGTH': 0.001,
                                                    'MIN_CANDLE_BODY_RATIO': 2.0,                                                
                                                    }
                                          }
                                            },
                     pred_label = 'hammer',
                     pred_type = 'symmetric',
                     pred_func= fn_pred3, 
                     pred_kwargs = {
                                    'indic': 'hammer',                                    
                                         'threshold_min': 1.5,
                                         'threshold_max': 999.9,
                                         'inv_threshold_fn': inv_fn_symmetric})

    c.run ()
if True:
    trades = []
    for i, t in enumerate(ds.p_df.index):
        signal = ds.p_df.Predictions[t]
        
        if signal != NEUTRAL_SIGNAL:
            trades.append (TradeSimulation (ds = ds, direction = signal,
                                            start_idx=i + 1,
                                            func_update_stop = fn_stop_update_trailing_v1,
                                            trailing_bars = 5,
                                            move_proportion = 0.7
                                            ))
            
    dummy = [trade.run_trade_by_one_bar () for trade in trades]
    valid_trades = [trade for trade in trades if trade.isValid]
    live_trades = [trade for trade in trades if trade.isAlive]
    
    while len (live_trades) > 0:
        dummy = [trade.run_trade_by_one_bar () for trade in live_trades]
        live_trades = [trade for trade in live_trades if trade.isAlive]
        
    trade = valid_trades[0]
    trade.plot (bTarget = False)
    
    ds.p_df['Aggregated MtM'] = np.zeros (len (ds.p_df))
    
    for trade in valid_trades:
        ds.p_df['Aggregated MtM'][trade.indices] += np.hstack((trade.mtms[0], np.diff(trade.mtms)))
    
    ds.p_df['Aggregated MtM'].cumsum ().plot ()