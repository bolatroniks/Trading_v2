# -*- coding: utf-8 -*-

import pandas as pd
from os import listdir
from os.path import isfile, join
import bintrees

from dateutil.relativedelta import relativedelta
from datetime import datetime

from Config.const_and_paths import *
from Framework.Miscellaneous.my_utils import *
from Framework.Dataset.Dataset import set_from_to_times
from Framework.Dataset.DatasetHolder import DatasetHolder

class PCA ():
    def __init__ (self, timeframe='M15', 
                  from_time=2000, 
                  to_time=None, 
                  instrument_list = full_instrument_list,
                  cov_window = 4 * 24 * 15, #number of observations used to compute covariance 
                  min_cut = 0.8,        #minimum percentage of valid observations in a timeseries for it to be considered
                  bSavePC = True, 
                  no_pc_to_save = 3, 
                  path = PCA_DEFAULT_PATH,
                  bLoadCandlesOnline = False
                  ):
        self.bLoadCandlesOnline = bLoadCandlesOnline
        self.timeframe = timeframe
        self.instrument_list = instrument_list
        self.from_time = None
        self.to_time = None
        set_from_to_times (self, from_time, 
                           (to_time if to_time is not None else str(datetime.today())[0:19]))
        self.dsh = DatasetHolder(from_time=self.from_time, 
                                 to_time=self.to_time,
                                 bLoadCandlesOnline = self.bLoadCandlesOnline)
        self.cov_window = cov_window
        self.min_cut = min_cut
        self.bSavePC = bSavePC
        self.no_pc_to_save = no_pc_to_save
        self.PCA_path = path
        
        
        
    def load_input_data (self):
        print ('############Loading time series###########################')
        self.dsh.loadMultiFrame(timeframe_list=[self.timeframe], 
                           ccy_pair_list=self.instrument_list, 
                           bComputeFeatures=[False], 
                           bComputeLabels=False, 
                           bLoadFeatures=[False], 
                           bLoadLabels=False)
        
        print ('############Merging indices###############################')
        full_idx = []
        for k, v in self.dsh.ds_dict.iteritems ():
            printProgressBar (10, 30, length=50,
                              suffix=str(k) + ': ' + str(len(full_idx)))
            
            if v.df is not None:
                if len(v.df) > 0:
                    dummy = map(full_idx.append, v.df.index)
                    full_idx = list(set(full_idx))
                    
        full_idx.sort()
        try:
            del dummy
        except:
            pass
        
        print ('############Building dataframe###########################')
        self.input_df = pd.core.frame.DataFrame(index = full_idx, 
                                    columns = self.dsh.ds_dict.keys ())
        
        for i, col in enumerate(self.input_df.columns):
            try:
                printProgressBar (i+1, len (self.input_df.columns),
                                  length=50,
                                  suffix=col)
                self.input_df[col] = self.dsh.ds_dict[col].df.Close[full_idx]
                del self.dsh.ds_dict[col]
                #m_df[col + '_rho_0'] = np.zeros (len(m_df))
                #m_df[col + '_residual_0'] = np.zeros (len(m_df))
            except:
                pass
            
    def extract_principal_components (self, bOverride = False):
        out_folder = join (self.PCA_path, 'PC', self.timeframe)
        
        pc_files = [f for f in listdir(out_folder) if isfile(join(out_folder, f))]
        
        pc_files_tree = bintrees.AVLTree(items=zip(pc_files,pc_files))
        
       #--------computes covariance matrix over a time window-------------#
        window = self.cov_window
        min_cut = self.min_cut
        m_df = self.input_df
        full_idx = self.input_df.index
        
        m_arr = np.array(m_df)
        print ('Extracting principal components')
        for i in range (len (m_df) - 1, window - 1, -1):
            if np.mod (i, 100) == 0:
                printProgressBar (len (m_df) - i + 1, len (m_df) - window, 
                                  suffix = 'Processing index ' + str (i+1) + '/' + str (len(m_df)),
                                  length=50)                
                
            if (str (m_df.index[i]) + '.csv') in pc_files_tree and not bOverride:
                continue
        #i=200000
        #if True:
            
                #plt.plot(resid_arr[i-1,:])
            #if True:
            try:
                sub_df = pd.core.frame.DataFrame(data=m_arr[i-window:i + 1,:], 
                                                 index=full_idx[i-window:i + 1],
                                                 columns=m_df.columns)
                col_idx = []
                
                for j, col in enumerate(m_df.columns):
                    if (m_df[col][i-window:i+1].dropna().shape[0] < min_cut * window) or (np.isnan(m_df[col].iat[-1])):
                        del sub_df[col]
                    else:
                        col_idx.append (j)
                        
                sub_df.dropna(inplace=True)
                arr = np.array(sub_df).astype(float)
                change = np.log(arr[1:, :] / arr[0:-1, :])
                M = np.dot(change.T, change)
                S,V, D = np.linalg.svd(M)
                self.PC = np.dot(change, S)
                
                if self.bSavePC:
                    self.save_principal_components (idx=sub_df.index)
            except Exception as e:
                printProgressBar (i+1-window, len (m_df) - window, 
                                  suffix = 'Error Processing index ' + str (i+1) + '/' + str (len(m_df)) + ' - ' + e.message,
                                  length=50)
            
    def save_principal_components (self, idx = None):
        filename = join (self.PCA_path, 'PC', self.timeframe, str (np.max(idx)) + '.csv')                
            
        sub_df = pd.core.frame.DataFrame(data=self.PC[:, :self.no_pc_to_save], 
                                                 index=idx[1:],
                                                 columns=['PC' + str(i) for i in range(self.no_pc_to_save)])
        sub_df.to_csv (filename)
        
    def load_principal_components (self, ts):
        filename = join (self.PCA_path, 'PC', self.timeframe, str (ts) + '.csv')
        
        self.pca_df = pd.read_csv(filename, parse_dates=[u'Unnamed: 0'], infer_datetime_format=True)
        self.pca_df.set_index (u'Unnamed: 0', inplace = True)
        self.pca_df.sort (inplace=True)
        
        return self.pca_df
    
    def compute_pca_features (self, ds, in_sample_cutoff = 0.75, 
                              bSave = True, bForceRecalc = True):
        ds.loadCandles ()
        
        out_filename =  ds.pca_feat_path + '/' +  \
                        ds.ccy_pair + '_' + ds.timeframe + '.csv'
        
        try:
            pca_df = ds.loadCsv2DF (ds.pca_feat_path, 
                         '',
                         '.csv',
                         ds.ccy_pair, ds.timeframe)
        except IOError:            
            outstr = ('Date,rho_0,resid_0,n_resid_0,rho_01,resid_01,n_resid_01,rho_012,resid_012,n_resid_012')
            f = open (out_filename, 'a')
            f.write (outstr + '\n')
            f.close ()
            pca_df = ds.loadCsv2DF (ds.pca_feat_path, 
                         '',
                         '.csv',
                         ds.ccy_pair, ds.timeframe)                
# =============================================================================
#             pca_df = pd.core.frame.DataFrame(index = [],
#                         columns = ['rho_0', 'resid_0', 'n_resid_0',
#                                    'rho_01', 'resid_01', 'n_resid_01',
#                                    'rho_012', 'resid_012', 'n_resid_012'])
# =============================================================================
        
        output_df = pd.core.frame.DataFrame(index = ds.df.index,
                        columns = ['rho_0', 'resid_0', 'n_resid_0',
                                   'rho_01', 'resid_01', 'n_resid_01',
                                   'rho_012', 'resid_012', 'n_resid_012'])
    
        old_start = ds.df.index[0]
        
        if not ds.bLoadCandlesOnline:
            ds.set_from_to_times(from_time=str(ds.df.index[0] - relativedelta(years=1)))
        
        ds.loadCandles ()
        
        sel = ['PC0', 'PC1', 'PC2']
        suffix = ['0', '01', '012']
        
        pca_df_index_tree = bintrees.AVLTree(items=zip(pca_df.index, pca_df.index))
        
        for i in range (len(output_df.index)):
            if (output_df.index [i] in pca_df_index_tree) and (not bForceRecalc):
                continue
            
            try:
                if np.mod (i,100) == 0:
                    printProgressBar( len(output_df.index) - i, len(output_df.index), prefix = 'Progress:', suffix = 'Complete', length = 50)
                df = self.load_principal_components(output_df.index[i])
                df['PC0_level'] = np.cumprod(1+df.PC0)
                df['PC1_level'] = np.cumprod(1+df.PC1)
                df['PC2_level'] = np.cumprod(1+df.PC2)
                df['Spot'] = ds.df.Close[df.index]
                #df.Spot -= df.Spot.mean ()                
                df['Change'] = ds.df.Change[df.index]
                df.dropna(inplace=True)
                
                if type (in_sample_cutoff) == float:
                    in_sample_cutoff_i = int(in_sample_cutoff * len(df.index))
                else:
                    in_sample_cutoff_i = in_sample_cutoff
                
                
                #denominator to normalize residual
                den = len(df) ** 0.5 * df.Change.std ()
                
                outstr = str(output_df.index [i])
                
                for j in range (3):
                    model = lr ()
                    model.fit_intercept = True                    
                    #x=np.array(df[['PC0_level', 'PC1_level', 'PC2_level']])
                    x=np.array(df[sel[0:j+1]])
                    #res = model.fit(X=x, y = df.Spot)
                    res = model.fit(X=x[0:in_sample_cutoff_i, :], 
                                    y = df.loc[df.index[0]:df.index[in_sample_cutoff_i-1]].Change)
                    df['pred_' + suffix[j]] = model.predict(x)
                    df['pred_level_' + suffix[j]] = df.Spot[0] * np.cumprod(df['pred_' + suffix[j]] + 1.0)
                    
                    rho = np.corrcoef(df.Change, df['pred_' + suffix[j]])[0,1]
                    output_df['rho_' + suffix[j]][output_df.index[i]] = rho
                    resid = df.Spot [-1] / df['pred_level_' + suffix[j]][-1] - 1.0
                    output_df['resid_' + suffix[j]][output_df.index[i]] = resid
                    n_resid = output_df['resid_' + suffix[j]][output_df.index[i]] / den
                    output_df['n_resid_' + suffix[j]][output_df.index[i]] = n_resid
                    
                    outstr += ',' + str (rho) + ',' + str (resid) + ',' + str (n_resid)
                
                if bSave:
                    f = open (out_filename, 'a')
                    f.write (outstr + '\n')
                    f.close ()
                
                printProgressBar (i+1, len(output_df.index), 
                                  length=50,
                                  prefix='Computing PCA features - ' + ds.ccy_pair + '_' + ds.timeframe,
                                  suffix = 'Calculated Successfully')
            except Exception as e:
                printProgressBar (i+1, len(output_df.index), 
                                  length=50,
                                  prefix='Computing PCA features - ' + ds.ccy_pair + '_' + ds.timeframe,
                                  suffix = e.message)
        
        ds.set_from_to_times(from_time=old_start)
        ds.loadCandles ()
        self.pca_feat_df = output_df
        return output_df
    
    def save_pca_features (self, ds):
        ds.saveDF2csv(self.pca_feat_df, 
                      r'/home/joanna/Desktop/Projects/Trading/datasets/Oanda/Fx/PCA_New', 
                      '', '.csv')
    
if False:
    from Framework.Dataset.Dataset import *
    ds = Dataset(ccy_pair='USD_ZAR', from_time='2013-01-01 00:00:00', to_time='2016-03-01 00:00:00', timeframe='M15')
    ds.loadCandles ()
    ds.computeFeatures ()
    ds.computeLabels(min_stop=0.02)
    
    pca = PCA(timeframe=ds.timeframe, from_time=ds.from_time, to_time=ds.to_time, instrument_list=full_instrument_list, bSavePC=False)
    
    IN_SAMPLE_CUTOFF = 800
    l = []
    for i in range (-19600,0,500):
        try:
            df = pca.load_principal_components(ds.df.index[i])
            df['PC0_level'] = np.cumprod(1+df.PC0)
            df['PC1_level'] = np.cumprod(1+df.PC1)
            df['PC2_level'] = np.cumprod(1+df.PC2)
            df['Spot'] = ds.df.Close[df.index]
            #df.Spot -= df.Spot.mean ()
            df['Change'] = ds.df.Change[df.index]
            df.dropna(inplace=True)
            
            from sklearn.linear_model import LinearRegression as lr
            
            
            model = lr ()
            model.fit_intercept = True
            sel = ['PC0', 'PC1', 'PC2']
            #x=np.array(df[['PC0_level', 'PC1_level', 'PC2_level']])
            x=np.array(df[sel])
            #res = model.fit(X=x, y = df.Spot)
            res = model.fit(X=x[0:IN_SAMPLE_CUTOFF, :], 
                            y = df.loc[df.index[0]:df.index[IN_SAMPLE_CUTOFF-1]].Change)
            df['pred'] = model.predict(x)
            #l.append (df.pred[ds.df.index[i]] - df.Spot[ds.df.index[i]])
            
            if True:
                fig = plt.figure ()
                plt.title (str(ds.df.index[i]) + ': ' + ['NEUTRAL', 'LONG', 'SHORT'][int(ds.l_df.Labels [ds.f_df.index[i]])])
                predSpot = np.cumprod(1+df.pred)
                plt.plot (predSpot * df.Spot[df.index[0]], label='pred')
                plt.plot(df.Spot, label='spot')
                plt.legend (loc='best')
                plt.show ()
        except:
            pass
        
if False:
    from Framework.Dataset.Dataset import Dataset
    
    if False:
        ds = Dataset(timeframe='D')
        for year in range (2017, 2004, -1):
            for instrument in random_list:#[-1:len(full_instrument_list)/2:-1]:
                try:
                    print ('Processing :' + instrument + ' ' + str (year))
                    ds.init_param (instrument = instrument,
                                   timeframe = 'D',
                                   from_time= year - 4, 
                                   to_time = year)
                    ds.loadCandles ()
                    ds.computeFeatures (bSaveFeatures=True,                          
                         bComputeIndicators=True,
                         bComputeNormalizedRatios=True,
                         bComputeCandles=True,
                         bComputeHighLowFeatures=True)
                except Exception as e:
                    print (e.message)
    
    if True:
        for year in range (2014, 2004, -1):
            for instrument in random_list:#[0:len(full_instrument_list)/2]:
                try:
                    ds = Dataset(ccy_pair=instrument, timeframe='M15', from_time=year, to_time=year)
                    pca = PCA(timeframe='M15', from_time=ds.from_time, to_time=ds.to_time, cov_window = 24 * 15 * 4)
                    pca.compute_pca_features(ds)
                except Exception as e:
                    print (e.message)                
            
if False:
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
            sub_df = pd.core.frame.DataFrame(data=m_arr[i-window:i+1,:], 
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