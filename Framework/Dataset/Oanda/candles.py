#!/usr/bin/env python

import sys

sys.path.append ('/home/Downloads/v20-python-samples-master/src')

import argparse
import common.config
import common.args
from instrument.view import CandlePrinter
from datetime import datetime


def get_candles(instrument='EURUSD', granularity='H1', from_time='2009-01-01 00:00:00', to_time='2010-01-01 00:00:00'):
    """
    Create an API context, and use it to fetch candles for an instrument.

    The configuration for the context is parsed from the config file provided
    as an argumentV
    """

    parser = argparse.ArgumentParser()

    #
    # The config object is initialized by the argument parser, and contains
    # the REST APID host, port, accountID, etc.
    #
    Trading.Execution.Broker.Oanda.common.config.add_argument(parser)

    parser.add_argument(
        "--instrument",
        type=Trading.Execution.Broker.Oanda.common.args.instrument,
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
        type=Trading.Execution.Broker.Oanda.common.args.date_time(),
        help="The start date for the candles to be fetched. Format is 'YYYY-MM-DD HH:MM:SS'"
    )

    parser.add_argument(
        "--to-time",
        default=to_time,
        type=Trading.Execution.Broker.Oanda.common.args.date_time(),
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

#if __name__ == "__main__":
my_candles = get_candles ()
