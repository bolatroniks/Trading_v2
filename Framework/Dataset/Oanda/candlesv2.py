#!/usr/bin/env python

import sys; sys.argv=['']
from Config.const_and_paths import *
import argparse

if V20_PATH not in sys.path:
    sys.path.append (V20_PATH)

if OANDA_PATH not in sys.path:
    sys.path.append (OANDA_PATH)



from instrument.view import CandlePrinter
#import common.config as config

import Framework.Dataset.Oanda.common.config as config
from Framework.Dataset.Oanda.common.args import instrument as args_instrument
from Framework.Dataset.Oanda.common.args import date_time
import datetime as dt
import numpy as np
import pandas

def print_candle(candle):
    try:
        time = str(
            datetime.strptime(
                candle.time,
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )
        )
    except:
        time = candle.time.split(".")[0]

    volume = candle.volume

    for price in ["mid", "bid", "ask"]:
        c = getattr(candle, price, None)

        if c is None:
            continue

        print("{:>{width[time]}} {:>{width[type]}} {:>{width[price]}} {:>{width[price]}} {:>{width[price]}} {:>{width[price]}} {:>{width[volume]}}".format(
            time,
            price,
            c.o,
            c.h,
            c.l,
            c.c,
            volume,
            width=candle.width
        ))

        volume = ""
        time = ""

def print_candles (candles):
    printer = CandlePrinter ()
    for candle in candles:
        print_candle(candle)

def saveDataframe(df, path='./datasets/Fx/Parsed/Oanda/Daily', filename=''):
    if filename == '':
        print ('Error: Need a name for the file')
        return
    df.to_csv(path+'/'+filename)
    
    
        
def loadCandlesIntoDataframe(candles):
    t_list = []
    o_array = np.zeros (len(candles))
    h_array = np.zeros (len(candles))
    l_array = np.zeros (len(candles))
    c_array = np.zeros (len(candles))
    v_array = np.zeros (len(candles))
    
    for i, candle in enumerate(candles):
        try:
            time = str(
                datetime.strptime(
                    candle.time,
                    "%Y-%m-%dT%H:%M:%S.000000000Z"
                )
            )
        except:
            time = candle.time.split(".")[0]
        t_list.append(time)
        volume = candle.volume
        v_array[i] = volume
    
        for price in ["mid", "bid", "ask"]:
            c = getattr(candle, price, None)
    
            if c is None:
                continue
            
            o_array[i] = c.o
            h_array[i] = c.h
            l_array[i] = c.l
            c_array[i] = c.c
    
#            print("{:>{width[time]}} {:>{width[type]}} {:>{width[price]}} {:>{width[price]}} {:>{width[price]}} {:>{width[price]}} {:>{width[volume]}}".format(
#                time,
#                price,
#                c.o,
#                c.h,
#                c.l,
#                c.c,
#                volume,
#                width=self.width
            #))
    
            volume = ""
            time = ""
    
    df = pandas.DataFrame ()
    df['Date'] = t_list
    df['Close'] = c_array
    df['Open'] = o_array
    df['High'] = h_array
    df['Low'] = l_array
    df['Volume'] = v_array
    
    #df['Volume'] = v_array

    a = np.zeros(len(c_array))
    a[1:]=c_array[1:] / c_array[0:-1] - 1
    
    a[0] = 0
    df['Change'] = a
    df['Date']  = pandas.to_datetime(df['Date'],infer_datetime_format =True)
    df.index = df['Date']
    df = df.sort_values(by='Date', ascending=True)
    del df['Date']
    
    return df

def get_candles(instrument='USD_ZAR', granularity='H1', from_time='', to_time='', default_n_periods=5000):
    """
    Create an API context, and use it to fetch candles for an instrument.

    The configuration for the context is parsed from the config file provided
    as an argumentV
    """
    print (granularity)
    if to_time == '':
        if granularity == 'H1':
            to_time = (dt.datetime.now () - dt.timedelta(hours=1) ).strftime ('%Y-%m-%d %H:%M:%S')
        elif granularity =='D':
            to_time = (dt.datetime.now () - dt.timedelta(hours=1) ).strftime ('%Y-%m-%d 00:00:00')
        print (to_time)
    if from_time == '':
        if granularity == 'H1':
            from_time = (dt.datetime.now () - dt.timedelta(hours=default_n_periods) ).strftime ('%Y-%m-%d %H:%M:%S')
        elif granularity =='D':
            from_time = (dt.datetime.now () - dt.timedelta(days=default_n_periods, hours=1) ).strftime ('%Y-%m-%d 00:00:00')
        print (from_time)
    parser = argparse.ArgumentParser()

    #
    # The config object is initialized by the argument parser, and contains
    # the REST APID host, port, accountID, etc.
    #
    config.add_argument(parser)
    #config.add_argument(parser)
    

    parser.add_argument(
        "--instrument",
        type=args_instrument,
        #type=args.instrument,
        default=instrument,
        help="The instrument to get candles for"
    )

    parser.add_argument(
        "--mid", 
        action='store_true',
        help="Get midpoint-based candles"
    )

    parser.add_argument(
        "--bid", 
        action='store_true',
        help="Get bid-based candles"
    )

    parser.add_argument(
        "--ask", 
        action='store_true',
        help="Get ask-based candles"
    )

    parser.add_argument(
        "--smooth", 
        action='store_true',
        help="'Smooth' the candles"
    )

    parser.set_defaults(mid=False, bid=False, ask=False)

    parser.add_argument(
        "--granularity",
        default=granularity,
        help="The candles granularity to fetch"
    )

    parser.add_argument(
        "--count",
        default=None,
        help="The number of candles to fetch"
    )

    date_format = "%Y-%m-%d %H:%M:%S"

    parser.add_argument(
        "--from-time",
        default=from_time,
        type=date_time(),
        #type=args.date_time(),
        help="The start date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
    )

    parser.add_argument(
        "--to-time",
        default=to_time,
        type=date_time(),
        #type=args.date_time(),
        help="The end date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
    )

    args = parser.parse_args()

    account_id = args.config.active_account

    #
    # The v20 config object creates the v20.Context for us based on the
    # contents of the config file.
    #
    api = args.config.create_context()

    kwargs = {}

    if args.granularity is not None:
        kwargs["granularity"] = args.granularity

    if args.smooth is not None:
        kwargs["smooth"] = args.smooth

    if args.count is not None:
        kwargs["count"] = args.count

    if args.from_time is not None:
        kwargs["fromTime"] = api.datetime_to_str(args.from_time)

    if args.to_time is not None:
        kwargs["toTime"] = api.datetime_to_str(args.to_time)

    price = "mid"

    if args.mid:
        kwargs["price"] = "M" + kwargs.get("price", "")
        price = "mid"

    if args.bid:
        kwargs["price"] = "B" + kwargs.get("price", "")
        price = "bid"

    if args.ask:
        kwargs["price"] = "A" + kwargs.get("price", "")
        price = "ask"

    #
    # Fetch the candles
    #
    response = api.instrument.candles(args.instrument, **kwargs)

    if response.status != 200:
        print(response)
        print(response.body)
        return

    print("Instrument: {}".format(response.get("instrument", 200)))
    print("Granularity: {}".format(response.get("granularity", 200)))

    printer = CandlePrinter()

    printer.print_header()

    candles = response.get("candles", 200)

    #for candle in response.get("candles", 200):
    #    printer.print_candle(candle)

    return candles

if False:
#if __name__ == "__main__":
    path='./datasets/Fx/Parsed/Oanda/Daily'
    instrument_list = ['AUD_JPY', 'AUD_USD', 'EUR_AUD', 'GBP_AUD', 'AUD_CHF',
                       'NZD_JPY', 'NZD_USD', 'EUR_NZD', 'GBP_NZD', 'NZD_CHF',
                       'ZAR_JPY', 'USD_ZAR', 'EUR_ZAR', 'GBP_ZAR', 'CHF_ZAR',
                       'MXN_JPY', 'USD_MXN', 'EUR_MXN', 'GBP_MXN', 'CHF_MXN',
                       'TRY_JPY', 'USD_TRY', 'EUR_TRY', 'GBP_TRY', 'CHF_TRY',]
    granularity = 'D'
    
    for instrument in instrument_list:
        try:
            filename = instrument+'-'+granularity+'.csv'
            
            my_candles = []
            df2 = pandas.core.frame.DataFrame ()
            
            #for i in range (2000, 2016, 1):
                #from_time = dt.datetime(i,11,21)
                #to_time = dt.datetime(i+1,11,21)
            
            my_candles = get_candles (instrument=instrument, granularity=granularity)
                
            df = loadCandlesIntoDataframe(my_candles)
                
            saveDataframe(df, path=path, filename=filename)
        except:
            print ('Error loading instrument: '+instrument)
