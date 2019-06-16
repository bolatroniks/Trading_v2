# -*- coding: utf-8 -*-

from Framework.Dataset.Dataset import *
import datetime as dt
from dateutil.relativedelta import relativedelta
from Config.const_and_paths import *

for ccy in full_instrument_list[20:]:
    ds = Dataset (ccy_pair=ccy, 
                  timeframe='M15', 
                  from_time=2000, 
                  to_time='2000-01-31 23:59:59')
    ds.initOnlineConfig ()
    
    from_time = dt.datetime(2000, 1,1,0,0,0)
    
    while from_time < dt.datetime.today ():
        to_time = from_time + relativedelta(months=1)
        
        ds.set_from_to_times(from_time = str(from_time), 
                             to_time = str(to_time))
        try:
            ds.loadSeriesOnline(bSaveCandles=True)
            print ('Loaded data succesfully for ' + str (from_time))
        except:
            print ('Failed to load data for ' + str (from_time))
        from_time = to_time
    
    #need to check if all dates were loaded
    ds.set_from_to_times(from_time=2002, to_time=2018)
    ds.loadCandles ()    
    full_dt_list = []
    from_time = dt.datetime(2002, 1,1,0,0,0)
    while from_time < dt.datetime.today ():
        if from_time.weekday() <5:
            full_dt_list.append(from_time)
        else: 
            print ('Trying to load missing data for ' + str (from_time))
            try:
                ds.loadSeriesOnline (from_time=str(from_time+relativedelta(days=-9)), 
                                 to_time=str (from_time + relativedelta(days=9)),
                                 bSaveCandles=True)
                print ('len(df): ' + str (len(ds.df)))
                from_time += relativedelta (days=7)
            except:
                print ('Failed to load data for ' + str (from_time))
        from_time += relativedelta (days=1)
    
    ds.loadCandles(from_time=2000, to_time=str(dt.datetime.today ()))
    dts_loaded = list(set([str(dt_str)[0:10] for dt_str in ds.df.index]))
    dts_checked = [str(my_date)[0:10] in dts_loaded for my_date in full_dt_list]
    plt.plot(full_dt_list, dts_checked)
    
if False:
    #this block checks whether labels have been computed
    #if not, it computes and saves them
    #could be interesting to implement the label computation in C
    for instrument in full_instrument_list[-1::-1]:
    try:
        ds = Dataset(ccy_pair=instrument, timeframe='M15', from_time=2000, to_time=2018)
        ds.loadLabels ()
        assert (ds.l_df.index[-1].year == 2018)
        
    except (IOError, AssertionError) as error:
        try:
            ds.set_from_to_times (from_time = ds.l_df.index[-1])
            ds.loadFeatures ()
            assert (ds.f_df.index[-1].year == 2018)
        except (AttributeError, AssertionError) as error:
            try:
                ds.loadCandles ()
                assert (ds.df.index[-1].year == 2018)
            except AssertionError:
                ds.download_oanda_data(since=ds.df.index[-1].year)
            except AttributeError:
                ds.download_oanda_data(since=2002)
            ds.set_from_to_times (from_time=2000)
            ds.loadCandles ()
            ds.computeFeatures ()
            ds.computeLabels (bSaveLabels=True)
    except:
        raise