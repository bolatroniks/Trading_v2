# -*- coding: utf-8 -*-

from Framework.Strategy.Strategy import *
from Framework.Execution.Broker import *
from Framework.Strategy.Rules import *

class Order ():
    def __init__ (self,   instrument=None,
                          direction='buy', 
                          units=1, orderType='Market',
                          signal_price = None, stop_loss_pct = None,
                          target_pct = None, status='open', 
                          strategy=None, broker=None, open_time=None,
                          termination_time=None, external_ID=None):
        
        self.instrument = instrument
        self.direction = direction
        self.units = units
        self.orderType = orderType
        self.signal_price = signal_price
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_px = None
        self.target_pct = target_pct
        self.target_px = None
        self.status = status
        self.strategy = strategy
        self.broker = broker
        self.open_time = open_time
        self.termination_time = termination_time
        self.external_ID = external_ID
        self.px_precision = 2
        
    def send (self):
        if self.broker is not None:
            self.validate ()
            self.broker.send_order (self)
            return True
        else:
            print ('Broker not chosen')
            return False
            
    def validate (self):
        print ('Order validation: To be implemented')
        pass
    
    def check_whether_executed (self):
        print ('Order check whether executed: To be implemented')
        return True        
        
            
        
        
        

