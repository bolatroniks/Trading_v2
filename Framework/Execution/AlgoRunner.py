# -*- coding: utf-8 -*-

from Framework.Strategy.Strategy import *
from Framework.Strategy.Strategy_PCA import *
from Framework.Execution.Broker.Broker import *
from Framework.Strategy.Rules.Rule import *
from Framework.Execution.Broker.Order import *
from Framework.Strategy.Rules.Production.MTF_PCA.multiframe_pca import *

from Framework.Reporting.Logging.LogManager import LogManager

import time
import numpy as np
from datetime import datetime as dt

class AlgoRunner ():
    def __init__ (self, name, 
                  strats = [], 
                  brokers = [], 
                  sleep_time=600, 
                  bVerbose=False):
        self.name = name
        self.strats = strats
        self.brokers = brokers
        self.sleep_time = sleep_time
        self.bVerbose = bVerbose
        
        self.log = LogManager.get_logger (self.name)
        self.log.info ('Algo Runner created')
        self.log.info ('Strategies: ' + str ([strat.name for strat in self.strats]))
        self.log.info ('Brokers: ' + str ([broker.name for broker in self.brokers]))
        self.log.info ('Sleep period: ' + str (self.sleep_time) )
        
        
    def run (self):
        self.reconcileAllPositions ()
        self.orders = []
        for strat in self.strats:
            strat.open_positions = self.overall_pos
            if True:
                self.log.info ('Starting to update signals for strategy: ' + strat.name)
                
                strat.updateSignals (last_timestamp = self.brokers[0].get_last_timestamp (), 
                                     instrument_list = strat.instruments)
                
                self.log.info ('Finished updating signals for strategy: ' + strat.name)
                
                orders, open_slots = strat.processSignals ()
                self.log.info ('Orders: ' + str (len(orders)) + ', slots: ' + str (open_slots))
                
                if open_slots < len(orders):
                    orders = orders [0:open_slots]
                    self.log.warning ('More orders than slots')
    
                self.orders += orders
                print ('Processing order: ' + str(orders))
                self.processOrders ()
                
                try:
                    while (dt.datetime.now ().time() > np.max([_.time() for _ in st.ds.df.index]) or
                           dt.datetime.now ().time() < np.min([_.time() for _ in st.ds.df.index])):
                        time.sleep (300)
                        LogManager.get_logger ().info ('Outside market hours, entering sleep mode')
                except Exception as e:
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)
        
        self.log.info ('Entering into sleep for ' + str (self.sleep_time) + ' seconds')
        time.sleep (self.sleep_time)
        
    def reconcileAllPositions (self):
        self.overall_pos = {}
        for broker in self.brokers:
            pos = broker.get_open_positions ()
            
            a = self.overall_pos
            b = pos
            
            r = dict(a.items() + b.items() +
                    [(k, a[k] + b[k]) for k in set(b) & set(a)])
            
            self.overall_pos = r
        self.log.info ('Overall positions: ' + str (self.overall_pos))
            
    def processOrders (self):
        print (self.orders)
        broker = self.brokers [0]
        for order in self.orders:
            px = broker.get_last_price (instrument=order.instrument)            
            
            if px['status'] == u'tradeable':
                bid_offer = float(px['asks'][0]['price']) / float(px['bids'][0]['price']) - 1.0
                mid_px = (float(px['asks'][0]['price']) + float(px['bids'][0]['price'])) / 2.0

                ask_px_str = px['asks'][0]['price']
                order.px_precision = np.maximum(len(ask_px_str[ask_px_str.find('.') + 1:]), 2)                

                if order.stop_loss_pct / bid_offer > 20.0:
                    if order.units > 0 and (float(px['asks'][0]['price']) / order.signal_price - 1.0) < order.stop_loss_pct / 20.0:
                        order.stop_loss_px = (1.0 - order.stop_loss_pct) * mid_px
                        order.target_px = (1.0 + order.target_pct) * mid_px
                        broker.send_order (order)
                    elif order.units < 0 and (float(px['bids'][0]['price']) / order.signal_price - 1.0) > - order.stop_loss_pct / 20.0:
                        order.stop_loss_px = (1.0 + order.stop_loss_pct) * mid_px
                        order.target_px = (1.0 - order.target_pct) * mid_px
                        broker.send_order (order)
        self.orders = []
        
if __name__ == "__main__":
    from Config.const_and_paths import CONFIG_PROD_RULE_PATH, CONFIG_LOG_PATH, V20_CONF, full_instrument_list
    from Framework.Strategy.Rules.Production.MTF_PCA.multiframe_pca import mtf_pca, mtf_pca_filter_slow, mtf_pca_filter_slow_and_fast
    from Framework.Strategy.Strategy_PCA import Strategy_PCA
    from Framework.Execution.Broker.Broker import Oanda
    import os
    from matplotlib import pyplot as plt
    
    filename = os.path.join(CONFIG_PROD_RULE_PATH, 'MTF_PCA', 'default.stp')
    f = open (filename, 'r')
    kwargs = eval(f.read ())
    
    kwargs['serial_gap'] = 0
    
    from Framework.Strategy.Rules.Rule import *
    rule=Rule(name='prod_mtf_pca', func = mtf_pca,
              filter_instrument_func = mtf_pca_filter_slow_and_fast,
              bUseHighLowFeatures = True,
              args=kwargs, ruleType='MultiTimeframe', 
              target_multiple=1.5)
    
    st = Strategy_PCA(name = 'PCA strategy - loose parameters',
                      rule=rule, 
                      instruments = full_instrument_list)
    broker = Oanda(my_config=V20_CONF)
    
    runner = AlgoRunner (name = 'default_strategy_loose',
                         strats=[st], 
                         brokers=[broker])
    
    runner.log.info ('Starting automatic execution')
    
    while True:
        try:
            runner.run ()
        except Exception as e:
            
            LogManager.get_logger ().error("Exception occurred", exc_info=True)
            time.sleep (600)
    
    if False:    
        start = time.time ()
        st.rule.filter_instrument_func = default_filter_instrument
        st.updateSignals (last_timestamp = broker.get_last_timestamp (),
                      instrument_list=['USD_ZAR'],
                      bComputePC=True)
    
        print ('Updated signals in ' + str (int(time.time () - start)) + ' seconds')

        for k in st.pred_dict.keys ():
            if st.pred_dict[k].dot(st.pred_dict[k]) != 0:
                if True:
                    fig = plt.figure ()
                    plt.title (k)
                    axes = plt.gca()
                    ax = axes.twinx ()
                    axes.plot(st.pred_dict[k], color='red')
                    st.ds.loadSeriesOnline (instrument=k)
                    ax.plot(st.ds.df['Close'], color='black')
                    plt.show ()

        