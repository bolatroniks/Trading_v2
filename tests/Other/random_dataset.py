# -*- coding: utf-8 -*-

from Framework.Training.VectorizedStrategy import *

#builds dataset of random timeseries, M15 candles and Daily features
if False:
    for i in range (6, 100):
        ds = Dataset(ccy_pair='EUR_HUF', timeframe='M15', from_time=2000, to_time=2018)
        ds.loadCandles ()
        
        ds.randomizeCandles ()
        ds.ccy_pair = 'USD_RANDOM' + str (i)
        ds.saveCandles ()
        
        ds2 = Dataset(ccy_pair=ds.ccy_pair, timeframe='D', 
                      from_time=ds.from_time, to_time=ds.to_time)
        ds2.buildCandlesFromLowerTimeframe(df=ds.df)
        ds2.saveCandles ()
        ds2.computeFeatures ()
        ds2.saveFeatures ()
        
        if False:
            for year in range (2017, 2004, -1):
                for instrument in ['USD_RANDOM' + str (i)]:#[0:len(full_instrument_list)/2]:
                    try:
                        ds = Dataset(ccy_pair=instrument, timeframe='M15', from_time=year, to_time=year)
                        pca = PCA(timeframe='M15', from_time=ds.from_time, to_time=ds.to_time, cov_window = 24 * 15 * 4)
                        pca.compute_pca_features(ds)
                    except Exception as e:
                        print (e.message)
#---------------------------------------------------------------

if True:
    bRandom = True
    output_file = r'/home/joanna/Desktop/Projects/Trading/tests/Other/Results/Strategy/default.csv'
    strategy_file = r'/home/joanna/Desktop/Projects/Trading/Trading/Training/GUI/Strats/default.stp'
    f = open(strategy_file, 'r')
    kwargs = eval(f.read ())
    f.close ()
    
    if bRandom:
        output_file = r'/home/joanna/Desktop/Projects/Trading/tests/Other/Results/Strategy/default_random.csv'
        kwargs['rho_max'] = 1.1
        kwargs['rho_min'] = -0.1
        kwargs['resid_max'] = 100.0
        kwargs['resid_min'] = -100.0
        
        instrument_list = random_list[0:15]
    else:
        instrument_list = full_instrument_list
    
    for year in range (2008, 2016):
        
        for instrument in instrument_list:
            try:
                vs = VectorizedStrategy (timeframe='M15', other_timeframes=['D'], from_time=year-2, to_time=year+2)
                vs.load_instrument(instrument=instrument, slow_timeframe_delay=0)
                set_from_to_times(vs, from_time=year, to_time=year+2)
                
                for serial_gap in [0, 100, 250, 500, 1000]:
                    kwargs['serial_gap'] = serial_gap
                    vs.preds_hash_table_dict = {}
                    vs.compute_pred_multiple(**kwargs)                    
                    
                    for min_stop in [0.5, 0.75, 1.0, 1.5, 2.0]:
                        for target_multiple in [1.0, 1.5, 2.0]:
                            try:
                                vs.hit_miss_cache = {}
                                vs.ds.computeLabels(min_stop=min_stop/100.0, 
                                                    target_multiple=target_multiple)
                                vs.compute_hit_miss ()
                                vs.summarize_stats ()
                                d = vs.strat_summary
                                del d['total_pnl']
                                del d['total_trades']
                                
                                key = d.keys() [0]
                                header_str = 'instrument'
                                outstr = key
                                for k, v in d[key].iteritems ():
                                    header_str += ',' + str (k)
                                    outstr += ',' + str(v)
                                outstr += ',' + str (serial_gap)+ '\n'
                                header_str += ',serial_gap\n'
                                
                                try:
                                    f = open(output_file, 'r')
                                    f.close ()
                                except IOError:
                                    f = open (output_file, 'a')
                                    f.write (header_str)
                                    f.close ()
                                f = open (output_file, 'a')
                                f.write (outstr)
                                f.close ()
                            except Exception as e:
                                for i in range (10):
                                    print e.message
                    
            except:
                pass