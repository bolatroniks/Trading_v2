from Trading.Dataset.DatasetHolder import *

ccy = 'EUR_SEK'
slow_timeframe = 'D'
fast_timeframe = 'M15'

dsh = DatasetHolder(from_time=2012, 
                    to_time=2015, instrument=ccy)
dsh.loadMultiFrame(timeframe_list=[slow_timeframe, fast_timeframe])


ds = dsh.ds_dict[ccy + '_' + fast_timeframe]
ds.computeFeatures (bComputeIndicators=True,
                             bComputeNormalizedRatios=True,
                             bComputeCandles=False,
                             bComputeHighLowFeatures=False)

dsh.appendTimeframesIntoOneDataset(instrument = ccy, 
                                   lower_timeframe = fast_timeframe,
                                   daily_delay=5)

ds.min_target = 0.0050
ds.vol_denominator = 20
ds.computeLabels ()

df = ds.f_df
df['trendlines_diff_10'] = df['no_standing_upward_lines_10_' + slow_timeframe] - df['no_standing_downward_lines_10_' + slow_timeframe]

if True:
    ds.l_df['Predictions'] = np.zeros(len(ds.l_df))
    #ds.l_df.Predictions[(df['RSI'] < 70) & (df['RSI_D'] > 50) & (df['trendlines_diff_10'] > 5)] = 1
    ds.l_df.Predictions[(df['RSI'] < 40) & (df['RSI_' + slow_timeframe] < 70) & (df['RSI_' + slow_timeframe ] > 50) & (df['trendlines_diff_10'] > 3)] = 1
    ds.l_df.Predictions[(df['RSI'] > 60) & (df['RSI_' + slow_timeframe] > 30) & (df['RSI_' + slow_timeframe] < 50) & (df['trendlines_diff_10'] < -3)] = -1

    fig = plt.figure ()
    axes = plt.gca()
    ax = axes.twinx ()
    axes.plot(ds.l_df.Predictions, color='red')
    ax.plot(ds.l_df.Close)
    
    fig = plt.figure ()
    plt.title ('PnL')
    plt.plot(np.cumsum(ds.l_df.Labels * ds.l_df.Predictions))
    plt.show ()