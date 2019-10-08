# -*- coding: utf-8 -*-

from Framework.Features.TimeSeries.hmm import *
from View import printProgressBar

from copy import deepcopy
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from Framework.Dataset.Dataset import Dataset
ds = Dataset(ccy_pair = 'USD_JPY', from_time =2000, to_time = 2019, timeframe = 'W')
ds.loadCandles ()
df = ds.df
year = 2000
training_period = 10
pred_delay = 0
prediction_threshold_margin = 0.5
no_states = 2

my_hmm = MyHMM (no_states = no_states,
                year_start_training = 2003,
                training_period = 10,
                xv_period = 5,
                pred_delay = 0,
                bRandomTest = False)

my_hmm.fit(df = df, cols = ['Change'], col_discriminator= 'Change', high_state_label = 'widen', low_state_label = 'tighten')  
my_hmm.predict ()
xv_df = deepcopy (my_hmm.xv_df)    

#risk parity strategy
bRiskParity = False
max_pos = 1.0
xv_df['vol'] = xv_df.Change.rolling (window=6).std ()
xv_df['Position'] = np.zeros (len(xv_df))
xv_df['Position'][xv_df[u'model_tighten_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = -1 * np.minimum(np.ones (len (xv_df)) * (0.02 / xv_df['vol'] if bRiskParity else 1.0), max_pos)
xv_df['Position'][xv_df[u'model_widen_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = 1 * np.minimum(np.ones (len (xv_df)) * (0.02 / xv_df['vol'] if bRiskParity else 1.0), max_pos)
xv_df['Change_fwd'] = (xv_df.Close.shift(-1) - xv_df.Close) * 100.0
xv_df['PnL'] = np.cumsum(xv_df['Change_fwd'] * xv_df.Position)


xv_df['Return'] = (xv_df.Close / xv_df.Close.shift(1) - 1)
#risk parity strategy
bRiskParity = True
max_pos = 1.0
xv_df['vol'] = xv_df.Return.rolling (window=6).std ()
xv_df['Position'] = np.zeros (len(xv_df))
xv_df['Position'][xv_df['model_' + my_hmm.low_state_label + '_prob'].shift(pred_delay) > np.maximum(0.5, (1.0 + prediction_threshold_margin) * (1.0 / no_states))] = -1 * np.minimum(np.ones (len (xv_df)) * (0.02 / xv_df['vol'] if bRiskParity else 1.0), max_pos)
xv_df['Position'][xv_df['model_' +my_hmm.high_state_label + '_prob'].shift(pred_delay) > np.maximum(0.5, (1.0 + prediction_threshold_margin) * (1.0 / no_states))] = 1 * np.minimum(np.ones (len (xv_df)) * (0.02 / xv_df['vol'] if bRiskParity else 1.0), max_pos)
xv_df['Ret_fwd'] = xv_df.Close.shift(-1) / xv_df.Close - 1.0
xv_df['PnL'] = np.cumsum(xv_df['Change_fwd'] * xv_df.Position)

fig = plt.figure (figsize=(14,8))
xv_df.PnL.plot (label='Strategy return')
(xv_df.Close/(xv_df.Close[0])).plot (label='SPX')
#np.cumprod(1 + spx_df['FEDFUNDS']/100.0/12).plot (label = 'risk free')
xv_df.Position.plot ()
plt.legend (loc='best')
plt.show ()

if True:
    #refactored SPX price only
    ret_list = []
    seeds = range (50)
    
    for bRandomTest in [False]:
        if bRandomTest:
            print ('Running random test')
        else:
            print ('Running on historical data')
        
        for i, seed in enumerate(seeds):
        #if True:
            no_states = 3
            year = 1960
            training_period = 30
            xv_period = 20
            pred_delay = 1
            claims_delay = 3
            
            np.random.seed (seed)
            prediction_threshold_margin = 0.5
                
            spx_filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/spx.csv'
            df = pd.read_csv (spx_filename, 
                                  parse_dates=['Date'], 
                                  index_col='Date', 
                                  infer_datetime_format=True).sort (ascending = True)
            
            
            df['Change'] = (df.SPX / df.SPX.shift(1) - 1)
            
            df.dropna (inplace = True)
            
            my_hmm = MyHMM (no_states = no_states,
                            year_start_training = year,
                            training_period = training_period,
                            xv_period = xv_period,
                            pred_delay = pred_delay,
                            bRandomTest = bRandomTest)
            
            my_hmm.fit(df = df, cols = ['Change'], col_discriminator= 'Change', high_state_label = 'bull', low_state_label = 'bear')  
            my_hmm.predict ()
            
            #loads risk free rate
            eff_filename = r'/home/joanna/Desktop/Projects/Trading/datasets/Macro/FEDFUNDS.csv'
            ff_df = pd.read_csv (eff_filename, 
                                  parse_dates=['DATE'], 
                                  index_col='DATE', 
                                  infer_datetime_format=True).sort(ascending=True)
            
            new_df = my_hmm.xv_df.join(ff_df, how='outer').dropna()
            
            new_df.SPX = np.cumproduct( 1 + new_df.Change)
            
            new_df['Return'] = (new_df.SPX / new_df.SPX.shift(1) - 1)
            #risk parity strategy
            bRiskParity = True
            max_pos = 1.0
            new_df['vol'] = new_df.Return.rolling (window=6).std ()
            new_df['Position'] = np.zeros (len(new_df))
            new_df['Position'][new_df[u'model_bear_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = -1 * np.minimum(np.ones (len (new_df)) * (0.02 / new_df['vol'] if bRiskParity else 1.0), max_pos)
            new_df['Position'][new_df[u'model_bull_prob'].shift(pred_delay) > (1.0 + prediction_threshold_margin) * (1.0 / no_states)] = 1 * np.minimum(np.ones (len (new_df)) * (0.02 / new_df['vol'] if bRiskParity else 1.0), max_pos)
            new_df['Ret_fwd'] = new_df.SPX.shift(-1) / new_df.SPX - 1.0
            new_df['PnL'] = np.cumprod(1 + new_df['FEDFUNDS']/100.0 * 1.0 / 12 + new_df['Position'] * (new_df['Ret_fwd']))
            new_df ['risk_free'] = np.cumprod(1 + new_df['FEDFUNDS']/100.0/12)
            
            if not bRandomTest:
                fig = plt.figure (figsize=(14,8))
                new_df.PnL.plot (label='Strategy return')
                (new_df.SPX/(new_df.SPX[0])).plot (label='SPX')            
                new_df.risk_free.plot (label = 'risk free')
                new_df.Position.plot ()
                plt.legend (loc='best')
                plt.show ()
                break
            
            ret_list.append (new_df.PnL[-2] / new_df.risk_free[-2])
            printProgressBar(i, len (seeds), length = 40)
        
        if bRandomTest:
            plt.hist (np.log(ret_list), bins = 50)
            
            print ('\nAvg return: ' + str (np.mean (ret_list) - 1))
            print ('Stdev return: ' + str (np.std (ret_list)))