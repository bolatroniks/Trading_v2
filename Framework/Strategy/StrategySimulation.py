# -*- coding: utf-8 -*-

#this simulates the full strategy, not only entry points
#takes into account stop and target functions

from Framework.Dataset.Dataset import *
from Framework.Genetic.Chromossome import *
from View import *
import numpy as np

global SIGNAL_CANCELLED
SIGNAL_CANCELLED = -999

#ToDo: need to deal with gaps as well as to come up with a model to estimate slippage
#ToDo: need a model to deal with transaction costs

#this function needs to return either:
#None: trade has not yet been entered
#SIGNAL_CANCELLED: signal is not longer valid
#price at which the trade has been entered
def fn_stop_entry (trade, **kwargs):
    signal_idx = trade.start_idx - 1
    
    trailing_bars = parse_kwargs (['trailing_bars_trigger_entry'], 0, **kwargs)
    kill_after = parse_kwargs (['kill_after'], 5, **kwargs)
    stop_entry_level = (trade.ds.p_df.High[signal_idx - trailing_bars:signal_idx + 1].max () \
                        if trade.direction == LONG_SIGNAL else \
                        trade.ds.p_df.Low[signal_idx - trailing_bars:signal_idx + 1].min ())
    try:
        t = trade.ts[-1]
    except:
        trade.ts = [trade.ds.p_df.index [trade.start_idx]]
    t = trade.ts[-1]
    #ToDo: check if there was no gap
    if stop_entry_level <= trade.ds.p_df.High[t] and trade.direction == LONG_SIGNAL:        
        return stop_entry_level
    if stop_entry_level >= trade.ds.p_df.Low[t] and trade.direction == SHORT_SIGNAL:
        return stop_entry_level
    
    if len (trade.events) >= kill_after:
        return SIGNAL_CANCELLED
    
    return None

#ToDo:
def fn_target_update_trailing_v1 (trade, **kwargs):
    return trade.targets [-1]

def fn_force_exit_n_bars (trade, **kwargs):
    n_bars = parse_kwargs (['n_bars'], None, **kwargs)
    
    if n_bars is None:
        return
    
    if n_bars <= len (trade.indices):
        return True
    

def fn_stop_init_v1 (trade, **kwargs):
    trailing_bars = parse_kwargs (['trailing_bars'], 5, **kwargs)
    trailing_low = np.min (trade.ds.p_df.Low[trade.start_idx-trailing_bars:trade.start_idx+1])
    trailing_high = np.max (trade.ds.p_df.High[trade.start_idx-trailing_bars:trade.start_idx+1])
    
    if trade.direction == LONG_SIGNAL:
        return trailing_low
    else:
        return trailing_high
    
def fn_target_init_v1 (trade, **kwargs):
    trailing_bars = parse_kwargs (['trailing_bars'], 5, **kwargs)
    trailing_low = np.min (trade.ds.p_df.Low[trade.start_idx-trailing_bars:trade.start_idx+1])
    trailing_high = np.max (trade.ds.p_df.High[trade.start_idx-trailing_bars:trade.start_idx+1])
    
    target_multiple = parse_kwargs (['target_multiple'], 2, **kwargs)
    
    if trade.direction == LONG_SIGNAL:
        return np.maximum(trailing_high, trade.start_px + target_multiple * ( trade.start_px - trade.stops[0]))
    else:
        return np.minimum(trailing_low, trade.start_px - target_multiple * ( trade.stops[0] - trade.start_px))

#ToDo: stop should also take into account a margin (absolue + fn(vol))
def fn_stop_update_trailing_v1 (trade, **kwargs):
    idx = trade.indices[-1]
    
    trailing_bars = parse_kwargs (['trailing_bars'], 5, **kwargs)
    move_proportion = parse_kwargs (['move_proportion'], 0.5, **kwargs)
    
    if trade.direction == LONG_SIGNAL:
        trailing_low = np.min (trade.ds.p_df.Low[idx-trailing_bars:idx+1])
        if trailing_low > trade.stops[-1]:
            return (trade.stops[-1] + move_proportion * (trailing_low - trade.stops[-1]))
        else:
            return trade.stops[-1]
    else:
        trailing_high = np.max (trade.ds.p_df.High[idx-trailing_bars:idx+1])
        if trailing_high < trade.stops[-1]:
            return (trade.stops[-1] + move_proportion * (trailing_high - trade.stops[-1]))
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
                        
        self.func_trigger_entry = parse_kwargs (['func_trigger_entry', 'fn_trigger_entry', 'fn_entry'], None, **kwargs)
        self.func_force_exit = parse_kwargs (['func_force_exit', 'fn_force_exit'], None, **kwargs)
        
        self.func_init_stop = func_init_stop = parse_kwargs (['func_init_stop'], None, **kwargs)
        self.func_init_target = func_init_target = parse_kwargs (['func_init_target'], None, **kwargs)
        self.func_update_stop = func_update_stop = parse_kwargs (['func_update_stop'], None, **kwargs)
        self.func_update_target = func_update_target = parse_kwargs (['func_update_target'], None, **kwargs)
        self.kwargs = kwargs
        
        self.func_d = {}
        self.kwargs_d = {}
        
        for k in ['trigger_entry', 'force_exit', 
                    'init_stop', 'init_target',
                    'update_stop', 'update_target']:
            v = parse_kwargs ([k], None, **kwargs)
            if v is not None:
                self.func_d [k] = parse_kwargs (['func'], None, **v)
                self.kwargs_d [k] = parse_kwargs (['kwargs'], None, **v)
                    
        self.open_lag = []
        self.high_lag = []
        self.low_lag = []
        self.close_lag = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.ts = []
        self.indices = []
        self.stops = []
        self.stops = []        
        
        self.isAlive = True
        self.isValid = True #checks whether trade not stopped before start
        self.mtms = []
        self.events = []
        
        self.handle_entry_point ()
        
    def exec_func (self, func_name):
        if func_name not in self.func_d.keys ():
            raise Exception ("Function not found")            
        
        if func_name in self.kwargs_d.keys ():
            kw = self.kwargs_d [func_name]
        else:
            kw = self.kwargs
        return self.func_d [func_name] (self, **kw)
        
    def handle_entry_point (self):
        try:            
            if self.isEntered:
                return   #the trade has already been entered into
        except:
            pass
        
        if self.start_t is None:
            if self.start_idx is not None:
                try:
                    self.start_t = self.ds.p_df.index[self.start_idx]
                except:
                    self.isValid = False
                    self.isAlive = False
            else:
                #ToDo: handle the case where only the timestamp is passed
                pass
        
        if self.func_trigger_entry is None and 'trigger_entry' not in self.func_d.keys (): #buys at the next bar's open price
            self.isEntered = True   #bool variable that tells whether the trade has already been triggered
                                    
            self.start_px = self.ds.p_df.Open[self.start_idx]
            self.ts = [self.start_t]
            self.indices = [self.start_idx]
        else:
            self.isEntered = False
            try:
                if self.func_trigger_entry is not None:
                    self.start_px = self.func_trigger_entry (self, **self.kwargs)
                elif 'trigger_entry' in self.func_d.keys ():                    
                    self.start_px = self.exec_func('trigger_entry')
            except:
                self.isAlive = False
                self.isValid = False
                return
            
            if self.start_px is not None: #the entry trigger has been hit
                if self.start_px == SIGNAL_CANCELLED:
                    self.isAlive = False
                    self.isValid = False
                    self.events.append ('Signal cancelled')
                else:
                    self.isEntered = True
            
        if self.isEntered:  #the trade has been triggered            
            if self.func_init_stop is None and 'init_stop' not in self.func_d.keys ():
                self.stops = ([self.ds.p_df.Low[self.start_idx - 1]] if self.direction == LONG_SIGNAL else [self.ds.p_df.High[self.start_idx - 1]])
            elif self.func_init_stop is not None:
                self.stops = [self.func_init_stop (self, **self.kwargs)]
            elif 'init_stop' in self.func_d.keys ():          
                self.stops = [self.exec_func('init_stop')]
            
            if self.func_init_target is None and 'init_target' not in self.func_d.keys ():
                self.targets = ([self.start_px + 2 * (self.start_px - self.stops[0])] if self.direction == LONG_SIGNAL else [self.start_px - 2 * np.abs(self.start_px - self.stops[0])])
            elif self.func_init_target is not None:
                self.targets = [self.func_init_target (self, **self.kwargs)]
            elif 'init_target' in self.func_d.keys ():                
                self.targets = [self.exec_func ('init_target')]
            
        if len (self.indices) > 0:
            self.indices += [self.indices[-1] + 1]
        else:
            self.indices = [self.start_idx]
        try:
            self.ts += [self.ds.p_df.index[self.indices[-1]]]
        except:
            self.isAlive = False
            self.isValid = False
        
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
        
        offset_x = len(self.events) - len (self.stops)
        fig = plot_candles (o=o, h=h, l=l, c=c, bShow = False)
        plt.plot (range (16 + offset_x, 16 + offset_x + len (self.stops) + 10), self.start_px * np.ones (len (self.stops) + 10), color='black', linestyle='--', label = 'entry')
        plt.plot (range (16 + offset_x, 16 + offset_x + len (self.stops)), self.stops, color='red', label = 'stop')
        if bTarget:
            plt.plot (range (16 + offset_x, 16 + offset_x + len (self.targets)), self.targets, color='green', label = 'targets')
        #plt.grid (True)
        plt.legend (loc='best')
        plt.show ()
        
    def run_trade_by_one_bar (self, bVerbose = False):
        if not self.isAlive:
            return
            
        if self.isEntered:
            #checks if trade not stopped before start
            #if func_trigger_entry is not None, this does not apply
            if (self.start_px <= self.stops[0] and self.direction == LONG_SIGNAL and self.func_trigger_entry is None) or \
                (self.start_px >= self.stops[0] and self.direction == SHORT_SIGNAL and self.func_trigger_entry is None):
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
            
            #checks whether trade should be exited by force
            if self.func_force_exit is not None or 'force_exit' in self.func_d.keys ():
                if self.func_force_exit is not None:
                    if self.func_force_exit (self, **self.kwargs):
                        #force exit
                        self.events.append ('Force exit')
                        self.isAlive = False
                        self.mtms.append ((self.ds.p_df.Close[self.ts[-1]] / self.start_px - 1) * self.quantity * (1 if self.direction == LONG_SIGNAL else -1))
                        return
                elif 'force_exit' in self.func_d.keys ():                    
                    if self.exec_func('force_exit'):
                        #force exit
                        self.events.append ('Force exit')
                        self.isAlive = False
                        self.mtms.append ((self.ds.p_df.Close[self.ts[-1]] / self.start_px - 1) * self.quantity * (1 if self.direction == LONG_SIGNAL else -1))
                        return
                else:
                    #nothing happens
                    pass
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
                self.stops.append (self.func_update_stop (self, **self.kwargs))
            elif 'update_stop' in self.func_d.keys ():
                self.stops.append (self.exec_func ('update_stop'))
            else:
                self.stops.append ( self.stops [-1] )
                
            if self.func_update_target is not None:
                self.targets.append (self.func_update_target (self, **self.kwargs))
            elif 'update_target' in self.func_d.keys ():
                self.targets.append (self.exec_func ('update_target'))
            else:
                self.targets.append ( self.targets [-1] )
            
            if bVerbose:
                print (self)
        else:    #trade has not yet been entered
            self.events.append ('Not entered yet')            
            self.handle_entry_point ()

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
    
class StrategySimulation ():
    def __init__ (self, ds = None, signals = None,
                  **kwargs
                  ):
        self.ds = ds
        
        if signals is not None:
            self.ds.set_predictions(signals)
        
        self.kwargs = kwargs
        
    def run (self, signals = None):
        if signals is not None:
            self.ds.set_predictions(signals)    
            
        try:
            assert(len(self.ds.p_df.Predictions) == len (self.ds.f_df))
        except Exception as e:
            raise e
        
        self.trades = []
        for i, t in enumerate(self.ds.p_df.index):
            signal = self.ds.p_df.Predictions[t]
            
            if signal != NEUTRAL_SIGNAL:
                self.trades.append (TradeSimulation (ds = self.ds, 
                                                direction = signal,
                                                start_idx=i + 1,
                                                **self.kwargs
                                                #func_update_stop = fn_stop_update_trailing_v1,
                                                #func_trigger_entry = fn_stop_entry,
                                                #trailing_bars_trigger_entry = 5,
                                                #kill_after = 3,
                                                #trailing_bars = 5,
                                                #move_proportion = 0.7
                                                ))
                
        dummy = [trade.run_trade_by_one_bar () for trade in self.trades]
        valid_trades = [trade for trade in self.trades if trade.isValid]
        live_trades = [trade for trade in valid_trades if trade.isAlive]
        
        len_live_trades_t0 = len (live_trades)
        while len (live_trades) > 0:
            dummy = [trade.run_trade_by_one_bar () for trade in live_trades]
            live_trades = [trade for trade in live_trades if trade.isAlive]
            printProgressBar(len_live_trades_t0 - len(live_trades), total = len_live_trades_t0)
        
        self.valid_trades = [trade for trade in self.trades if trade.isValid]
                
        trade = valid_trades[0]
        trade.plot (bTarget = False)
        
        aggregated_mtm = np.zeros (len (self.ds.p_df))
        
        for trade in [_ for _ in self.valid_trades if _.isValid]:
            try:
                aggregated_mtm[trade.indices[-len (trade.mtms):]] += np.hstack((trade.mtms[0], np.diff(trade.mtms)))
            except:
                pass
        
        self.ds.p_df['Aggregated MtM'] = aggregated_mtm
        
    def diagnostics (self, bVerbose = False, bShowPlots = False):
        mtms_0 = [trade.mtms[0] for trade in self.valid_trades]
        mtms = [trade.mtms[-1] for trade in self.valid_trades]
        cum_ret = self.ds.p_df['Aggregated MtM'].cumsum()[-1]
        delta_t = (self.ds.p_df.index[-1] - self.ds.p_df.index[0]).days / 365.25
        sigma = self.ds.p_df['Aggregated MtM'].std () * (delta_t * 252 )** 0.5
        
        self.stats = {'no_bars': len(self.ds.p_df),
                      'no_signals' : len(self.trades),
                      'no_valid_trades' : len(self.valid_trades),
                      'mean_mtm_0' : np.mean(mtms_0),
                      'std_mtm_0' : np.std(mtms_0),
                      'count_mtm_0_positive' : np.count_nonzero(np.array (mtms_0) > 0),
                      'count_mtm_0_negative' : np.count_nonzero(np.array (mtms_0) < 0),
                      'mean_mtm_T' : np.mean(mtms),
                      'std_mtm_T' : np.std(mtms),
                      'count_mtm_T_positive' : np.count_nonzero(np.array (mtms) > 0),
                      'count_mtm_T_negative' : np.count_nonzero(np.array (mtms) < 0),
                      'cum_ret' : cum_ret,
                      'sigma': sigma,
                      'sharpe': cum_ret / sigma,
                      't_start': str(self.ds.p_df.index[0]),
                      't_end': str(self.ds.p_df.index[-1]),
                     }
        
        if bShowPlots:
            fig = plt.figure ()
            plt.title ('Strat aggregated MtM ' + self.ds.ccy_pair)
            self.ds.p_df['Aggregated MtM'].cumsum ().plot ()
            plt.show ()
            
            fig = plt.figure ()
            plt.title ('PnL per trade histogram')
            
            plt.hist (mtms, bins = 100)
            plt.show ()
            
        if bVerbose:
            print ('######----Trade statistics-----########')
            print ('Number of bars: ' + str (len(self.ds.p_df)))
            print ('Signals: ' + str (len(self.trades)))
            print ('Valid trades: ' + str (len(self.valid_trades)))
            print ('Mean MtM(0): ' + str (np.mean(mtms_0)) )
            print ('Std MtM(0): ' + str (np.std(mtms_0)))
            print ('count MtM(0) < 0 - ' + str (np.count_nonzero(np.array (mtms_0) < 0)))
            print ('count MtM(0) > 0 - ' + str (np.count_nonzero(np.array (mtms_0) > 0)))
            print ('Mean MtM(t): ' + str (np.mean(mtms)) )
            print ('Std MtM(t): ' + str (np.std(mtms)))
            print ('count MtM(t) < 0 - ' + str (np.count_nonzero(np.array (mtms) < 0)))
            print ('count MtM(t) > 0 - ' + str (np.count_nonzero(np.array (mtms) > 0)))
            print ('Cumulative return: ' + str (self.ds.p_df['Aggregated MtM'].cumsum()[-1]))
        
        return self.stats

#generates signals    
if False:
    timeframe = 'M15'
    ds = Dataset(ccy_pair='SPX500_USD', 
                              from_time = 2000,
                              to_time=2013, 
                              timeframe = timeframe)
    ds.loadCandles ()
    ds.computeFeatures ()
    #creates a chromossome
    c = Chromossome (ds = ds, bDebug = True, bSlim = False)

    #RSI momentum
    if True:
        c.add_gene(
                         gene_id = 'RSI_momentum_symmetrical',
                         timeframe = 'D',
                         pre_type = 'symmetric',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 53,
                                             'threshold_max': 70,
                                             'inv_threshold_fn': inv_fn_rsi})
    
    c.add_gene (timeframe = 'D', func_dict = {'new_hilo':{'func':fn_new_hilo, 
                                                          'kwargs':{'window': 252,
                                                                    'conv_window': 25}
                                                          }
                                            },
                                pred_label = 'new_hilo',
                                pred_type = 'symmetric',
                                pred_func = fn_pred3,
                                pred_kwargs = {
                                                'indic': 'new_hilo',
                                                'threshold_min': 0.5, 
                                                'threshold_max': 1.5,
                                                'inv_threshold_fn': inv_fn_symmetric
                                            })


    if False:
        c.add_gene(
                         gene_id = 'RSI_oversold_symmetrical',
                         timeframe = 'M15',
                         pre_type = 'symmetric',
                         pred_label = 'RSI',
                         pred_func= fn_pred3, 
                         pred_kwargs = {'indic': 'RSI',
                                             'threshold_min': 30,
                                             'threshold_max': 35,
                                             'inv_threshold_fn': inv_fn_rsi})


    if True:
        c.add_gene(timeframe = timeframe,
                     func_dict = {'hammer':{'func':fn_hammer, 
                                          'kwargs':{'conv_window': 10,
                                                    'MIN_BAR_LENGTH': 0.001,
                                                    'MIN_CANDLE_BODY_RATIO': 2.5,                                                
                                                    }
                                          }
                                            },
                     pred_label = 'hammer',
                     pred_type = 'symmetric',
                     pred_func= fn_pred3, 
                     pred_kwargs = {
                                    'indic': 'hammer',                                    
                                         'threshold_min': 0.5,
                                         'threshold_max': 999,
                                         'inv_threshold_fn': inv_fn_symmetric})

    c.run ()
    #c.ds.removeSerialPredictions (10)

    kwargs = {'func_update_stop': fn_stop_update_trailing_v1,
            'func_trigger_entry' : fn_stop_entry,
            'trailing_bars_trigger_entry' : 5,
            'kill_after' : 3,
            'trailing_bars' : 5,
            'move_proportion' : 0.7}
    strat = StrategySimulation (ds = c.ds, signals = None, **kwargs)
    strat.run ()