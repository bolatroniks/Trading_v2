# -*- coding: utf-8 -*-

import argparse
import commands

from Logging import LogManager
from Framework.Dataset import config, args as common_args
import time
import numpy as np
import os

#---------These must be saved in a config file----------#
from Config.const_and_paths import *

class Broker ():
    def __init__ (self):
        self.name = 'Generic_Broker'
        pass

class Oanda (Broker):
    def __init__ (self, my_config = V20_CONF):
        self.config = my_config
        parser = argparse.ArgumentParser()
        config.add_argument(parser)
        args = parser.parse_args(args = ['--config', self.config])        
        self.account_id = args.config.active_account        
        
        self.api = args.config.create_context()
        
        self.last_timestamp = None
        self.positions = None
        self.orders = None
        self.name = 'Oanda'
        
    def send_order (self, order):
        cwd = os.getcwd()
        os.chdir (OANDA_PATH)
        #os.system( '/home/renato/Downloads/v20-python-samples-master/src/order/limit.py', wdir='/home/renato/Downloads/v20-python-samples-master', args='--instrument AUD_NZD --units 5000 --price 1.10 --time-in-force GTC')
        
        order_str = 'python ./order/' + OANDA_MARKET_ORDER_SCRIPT + \
                  ' --config ' + self.config + \
                  ' --time-in-force FOK ' + \
                   ' --take-profit-price ' + str(np.round(order.target_px, order.px_precision)) + \
                   ' --stop-loss-price ' + str (np.round(order.stop_loss_px, order.px_precision)) +\
                   ' ' + order.instrument + \
                  ' ' + str(order.units)

        print ('Order: ' + order_str)        
        status, output = commands.getstatusoutput(order_str)
        LogManager.get_logger().info (str (status) + ' - ' + str (output))
        os.chdir (cwd)
        
    def get_sent_orders (self):
        response = self.api.order.list(self.account_id)
        self.orders = response.get("orders", 200)    #in Oanda format, at some point should create order objects
        
        
    def get_open_positions (self):
        response1 = self.api.position.list_open (self.account_id)
        print ('Step 1 successful')
        response2 = response1.get("positions", 200)
        print ('Step 2 successful')
        
        self.positions = {}
        for pos in response2:
            self.positions[str(pos.instrument)] = float(pos.long.dict ()['units']) + float(pos.short.dict ()['units'])
        
        print ('Step 3 successful')
        return self.positions
    
    def get_last_timestamp (self):
        px = self.get_last_price ()
        return self.last_timestamp
        
    def get_transactions (self, fromid=1, toid=50, transaction_filter=None):
        response = self.api.transaction.range(
                                         self.account_id,
                                         fromID=fromid,
                                         toID=toid,
                                         type=transaction_filter)
        self.transactions = response.get("transactions", 200)
        
        self.last_transaction_id = self.transactions[-1].dict ()['id']

        if self.last_transaction_id == str(toid):
            print ('Recursive call')
            self.get_transactions (fromid=fromid, toid=toid+10, 
                                   transaction_filter=transaction_filter)
        
    def get_last_price (self, instrument='EUR_USD'):
        parser = argparse.ArgumentParser()
        
        config.add_argument(parser)
        
        parser.add_argument(
                '--instrument',
                type=common_args.instrument,
                required=True,
                action="append",
                help="Instrument to get prices for"
            )
        
        parser.add_argument(
                '--poll',
                action="store_true",
                default=False,
                help="Flag used to poll repeatedly for price updates"
            )
        
        parser.add_argument(
                '--poll-interval',
                type=float,
                default=2,
                help="The interval between polls. Only relevant polling is enabled"
            )
        
        args = parser.parse_args(args = ['--instrument', instrument, '--config', self.config])
        
        #account_id = args.config.active_account
        
        self.api = args.config.create_context()
        
        latest_price_time = None


        """
        Fetch and display all prices since than the latest price time

        Args:
            latest_price_time: The time of the newest Price that has been seen

        Returns:
            The updated latest price time
        """

        response = self.api.pricing.get(
            self.account_id,
            instruments=",".join(args.instrument),
            since=latest_price_time,
            includeUnitsAvailable=False
        )

        #
        # Print out all prices newer than the lastest time 
        # seen in a price
        #
        for price in response.get("prices", 200):
            if latest_price_time is None or price.time > latest_price_time:
                try:
                    print(view.price_to_string(price))
                except:
                    pass

        #
        # Stash and return the current latest price time
        #
        for price in response.get("prices", 200):
            if latest_price_time is None or price.time > latest_price_time:
                latest_price_time = price.time
                
        self.last_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.strptime(latest_price_time[0:19], 
                                                          '%Y-%m-%dT%H:%M:%S'))
        
        return price.dict ()

if False:
    broker = Oanda (my_config=V20_CONF)    
    ts = broker.get_last_timestamp ()
    dt = parser.parse(ts)