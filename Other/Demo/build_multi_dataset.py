# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy
from scipy import stats

from Framework.Dataset.DatasetHolder import *

timeframe = 'M15'

dsh = DatasetHolder(from_time=2000, to_time=2018)
print ('############Loading time series###########################')
dsh.loadMultiFrame(timeframe_list=[timeframe], ccy_pair_list=full_instrument_list, bComputeFeatures=[False], bComputeLabels=False, bLoadFeatures=[False], bLoadLabels=False)

print ('############Merging indices###############################')
full_idx = []
for k, v in dsh.ds_dict.iteritems ():
    print str(k) + ': ' + str(len(full_idx))
    if v.df is not None:
        if len(v.df) > 0:
            dummy = map(full_idx.append, v.df.index)
            full_idx = list(set(full_idx))
            
full_idx.sort()

print ('############Building dataframe###########################')
m_df = pd.core.frame.DataFrame(index = full_idx, 
                            columns = dsh.ds_dict.keys ())

for col in m_df.columns:
    try:
        print col
        m_df[col] = dsh.ds_dict[col].df.Close[full_idx]
        #m_df[col + '_rho_0'] = np.zeros (len(m_df))
        #m_df[col + '_residual_0'] = np.zeros (len(m_df))
    except:
        pass

#--------computes covariance matrix over a time window-------------#
window = 4 * 24 * 15
min_cut = 0.8

m_arr = np.array(m_df)
rho_arr = np.zeros ((len (m_df), len(m_df.columns)))
r_value_arr = np.zeros ((len (m_df), len(m_df.columns)))
p_value_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_arr = np.zeros ((len (m_df), len(m_df.columns)))
n_resid_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_1_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_5_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_10_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_90_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_95_arr = np.zeros ((len (m_df), len(m_df.columns)))
resid_99_arr = np.zeros ((len (m_df), len(m_df.columns)))

for i in range (window, len (m_df)):
#i=200000
#if True:
    if np.mod (i, 100) == 0:
        print ('Processing index ' + str (i+1) + '/' + str (len(m_df)))
        plt.plot(resid_arr[i-1,:])
    #if True:
    try:
        sub_df = pd.core.frame.DataFrame(data=m_arr[i-window:i,:], 
                                         index=full_idx[i-window:i],
                                         columns=m_df.columns)
        col_idx = []
        
        for j, col in enumerate(m_df.columns):
            if m_df[col][i-window:i].dropna().shape[0] < min_cut * window:
                del sub_df[col]
            else:
                col_idx.append (j)
                
        sub_df.dropna(inplace=True)
        arr = np.array(sub_df).astype(float)
        change = arr[1:, :] / arr[0:-1, :] - 1.0
        M = np.dot(change.T, change)
        S,V, D = np.linalg.svd(M)
        PC = np.dot(change, S)
        
        for j, k in enumerate(col_idx):
            rho_arr[i][k] = np.corrcoef(PC[:, 0], change[:, j])[0,1]
            
            slope, intercept, r_value_arr[i, j], p_value_arr[i,j], std_err = stats.linregress(x=np.exp(np.cumsum(PC[:,0])),
                                                                           y=np.exp(np.cumsum(change[:,j])))
            
            pred_values = slope * np.exp(np.cumsum(PC[:,0])) + intercept
            resid = np.exp(np.cumsum(change[:, j])) - pred_values
            resid_arr[i,k] = resid[-1]
            n_resid_arr[i,k] = resid[-1] / change[:,j].std ()
            resid_1_arr[i,k] = np.percentile(resid, 1)
            resid_5_arr[i,k] = np.percentile(resid, 5)
            resid_10_arr[i,k] = np.percentile(resid, 10)
            resid_90_arr[i,k] = np.percentile(resid, 90)
            resid_95_arr[i,k] = np.percentile(resid, 95)
            resid_99_arr[i,k] = np.percentile(resid, 99)
            
            
            
            if False:
                fig = plt.figure ()
                plt.title (str (r_value_arr[i, j]) +', ' + str (rho_arr[i][j]))
                #plt.plot(pred_values, label='preds')
                #plt.plot(np.exp(np.cumsum(change[:,j])), label='obs')
                plt.plot (resid)
                plt.legend (loc='bottom')
                plt.show ()
    except:
        pass
#------------------------------------------------------------------#
#--------adds new features to dataframe----------------------------#
for j, col in enumerate(m_df.columns):
    try:
        m_df[col+'_rho'] = rho_arr[:,j]
        m_df[col+'_r_value'] = r_value_arr[:,j]
        m_df[col+'_p_value'] = p_value_arr[:,j]
        m_df[col+'_resid'] = resid_arr [:, j]
        m_df[col+'_n_resid'] = n_resid_arr [:, j]
        m_df[col+'_resid_1_quantile'] = resid_1_arr [:, j]
        m_df[col+'_resid_5_quantile'] = resid_5_arr [:, j]
        m_df[col+'_resid_10_quantile'] = resid_10_arr [:, j]
        m_df[col+'_resid_90_quantile'] = resid_90_arr [:, j]
        m_df[col+'_resid_95_quantile'] = resid_95_arr [:, j]
        m_df[col+'_resid_99_quantile'] = resid_99_arr [:, j]                
    except:
        pass
    
#--------saves timeframe------------------------------------------#
full_filename = r'./datasets/Oanda/Fx/PCA/' + str (timeframe) + '.csv'
#m_df.to_csv (full_filename)

for ccy in full_instrument_list:
    sub_df = m_df[[col for col in m_df.columns if col.find(ccy) >= 0]]
    print ('Saving ' + ccy)
    sub_df.to_csv (r'./datasets/Oanda/Fx/PCA/' + ccy + '_' +  str (timeframe) + '.csv')

if False:       
    df_complete = m_df
    #check if file already exists
    try:
        df2 = pd.read_csv (full_filename)
        df3 = df2.reset_index().merge(m_df.reset_index(), how='outer').set_index('Date')
        #df_complete = df3.reset_index().drop_duplicates(subset='Date', keep='last').set_index('Date')
        
        df_complete = df3[~df3.index.duplicated(keep='last')]
        try:
            del df_complete['index']
        except:
            pass
        df_complete = indexDf (df_complete)
    except:
        pass
    df_complete.to_csv (full_filename)