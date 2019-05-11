
# -*- coding: utf-8 -*-

import numpy as np
import pandas
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime

try:
    from matplotlib import pyplot as plt
except:
    pass

def p2f(x):
    try:
        y = float(x.strip('%'))/100
    except:
        print ('Error converting to float:'+str(x))
        y = 0
    return y

def plotSeriesByNo (series_no, path="./", filename_prefix='ccy_hist_feat_', filetype='label', field='Close', df_sel='cv'):
    if True:
    #try:
        my_df = loadSeriesToDataframe (path, filename_prefix+str(series_no)+'.csv')
        train_df, cv_df, test_df = splitDataframeIntoTrainCVTest (my_df)
        if df_sel == 'cv':
            plt.plot(cv_df[field])
        elif df_sel == 'test':
            plt.plot(test_df[field])
    #except:
    #    print ("Error loading dataframe")
    
def splitDataframeIntoTrainCVTest (df, n_cv=1000, n_test=1000, bLoadOnlyTrainset=False):
    #check if df length is enough
    if len(df) < n_cv + n_test:
        print ("Dataframe too small")
    
    print ('Len df: '+str(len(df)))
        
    train_df = df [0:-n_cv-n_test]
    if bLoadOnlyTrainset == False:
        cv_df = df[-n_test-n_cv:-n_test]
        test_df = df[-n_test:]  
        print ('Should load correctly')
    else:
        cv_df = [1]
        test_df = [1]
    #print ('Splitting len cv_df: '+str(len(cv_df)))

    return train_df, cv_df, test_df

def buildSentencesOnTheFly (feat_df):
    
    #labels_df2 = labels_df['Labels']
    #labels_df = pandas.core.frame.DataFrame()
    #labels_df['Labels'] = labels_df2
    #feat_df = feat_df.merge(labels_df)
    #feat_df = pandas.concat([feat_df, labels_df], axis=1)
    #feat_df = feat_df.dropna()
    #labels_df = feat_df.ix[:,'Labels':]
    #del feat_df ['Labels']
    
    #feat_no = np.shape(np.array(feat_df.ix[:,'Close':].irow(0).values))[0]
    #feat_no = np.shape(np.array(feat_df.irow(0).values))[0]
    feat_no = np.shape(np.array(feat_df.iloc[0].values))[0]
    sentences = np.zeros ((len(feat_df),feat_no))
    #next_chars = np.zeros (len(feat_df))

    #if bZeroLongDatedMovingAverages == True:
    #    feat_df['ma_200_close'] = 0
    #    feat_df['ma_100_close'] = 0
    for i in range(len(feat_df)):
        #sentences[i,:] = np.array(feat_df.ix[:,'Close':].\irow(i).values)
        #sentences[i,:] = np.array(feat_df.irow(i).values)
        sentences[i,:] = np.array(feat_df.iloc[i].values)
        #next_chars [i] = labels_df ['Labels'].irow(i)
    return sentences#, next_chars
    
    
def buildSentences (feat_df, labels_df, bZeroLongDatedMovingAverages=False):
    
    if labels_df is not None:
        labels_df2 = labels_df['Labels']
        labels_df = pandas.core.frame.DataFrame()
        labels_df['Labels'] = labels_df2
        #feat_df = feat_df.merge(labels_df)
        feat_df = pandas.concat([feat_df, labels_df], axis=1)
        feat_df = feat_df.dropna()
        labels_df = feat_df.ix[:,'Labels':]
        del feat_df ['Labels']
        #sentences = np.zeros ((len(feat_df),len(feat_df.columns)-1))
        #print('Route 1:'+str(len(feat_df.columns)))
    else:
        labels_df = pandas.core.frame.DataFrame()
        labels_df['Labels'] = feat_df['Close']

        feat_df = pandas.concat([feat_df, labels_df], axis=1)
        feat_df = feat_df.dropna()
        labels_df = feat_df.ix[:,'Labels':]
        del feat_df ['Labels']
        #sentences = np.zeros ((len(feat_df),len(feat_df.columns)-1))
        #print('Route 2:'+str(len(feat_df.columns)))
    #print ("Dataframe Columns: "+ str(feat_df.columns))
 
    sentences = np.zeros ((len(feat_df),len(feat_df.ix[:,'Close':].columns)))
    
    next_chars = np.zeros (len(feat_df))

    if bZeroLongDatedMovingAverages == True:
        feat_df['ma_200_close'] = 0
        feat_df['ma_100_close'] = 0
    for i in range(len(feat_df)):
        sentences[i,:] = np.array(feat_df.ix[:,'Close':].irow(i).values)
        
        next_chars [i] = labels_df ['Labels'].irow(i)
    return sentences, next_chars

def buildSequencePatches (sentences, next_signal, lookback_window=252, no_signals=3):
    #print ('Shape sentences: '+str(np.shape(sentences)))
    #print ('Shape next signal: '+str(np.shape(next_signal)))
    n_features = np.size (sentences, 1)
    #print ('sentences: '+str(len(sentences)))
    #try:
    X = np.zeros((len(sentences)-lookback_window+1, lookback_window, n_features))
    y = np.zeros((len(sentences)-lookback_window+1,no_signals), dtype=np.bool)
    
    #test_X = np.zeros((len(test_sentences)-lookback_window+1, lookback_window, n_features))
    #test_y = np.zeros((len(test_sentences)-lookback_window+1,len(chars)), dtype=np.bool)
    
    #----------builds train_X and train_Y 
    for i in range(len(sentences)-lookback_window+1):
        for j in range (lookback_window):
            X[i,j,:] = sentences[i+j,:]
        #next_label = ''        

        if next_signal[i+lookback_window-1] == -1:
            y[i,:] = [1,0,0]
        elif next_signal[i+lookback_window-1] == 0:
            y[i,:] = [0,1,0]
        elif next_signal[i+lookback_window-1] == 1:
            y[i,:] = [0,0,1]
    #except:
    #    X=[]
    #    y=[]
    return X, y

def buildSequencePatchesOnTheFly (sentences, lookback_window=252, no_signals=3):
    n_features = np.size (sentences, 1)
    #print ('sentences: '+str(len(sentences)))
    X = np.zeros((len(sentences)-lookback_window+1, lookback_window, n_features))
    #y = np.zeros((len(sentences)-lookback_window+1,no_signals), dtype=np.bool)
    
    #test_X = np.zeros((len(test_sentences)-lookback_window+1, lookback_window, n_features))
    #test_y = np.zeros((len(test_sentences)-lookback_window+1,len(chars)), dtype=np.bool)
    
    #----------builds train_X and train_Y 
    for i in range(len(sentences)-lookback_window+1):
        for j in range (lookback_window):
            X[i,j,:] = sentences[i+j,:]
        
    #    if next_signal[i] == -1:
    #        y[i,:] = [1,0,0]
    #    elif next_signal[i] == 0:
    #        y[i,:] = [0,1,0]
    #    elif next_signal[i] == 1:
    #        y[i,:] = [0,0,1]
    return X#, y

def loadFeaturesNames (path, filename, other_feats_list=['./datasets/Macro/vix2.csv']):
    my_df = pandas.read_csv(path+'/'+filename, nrows=5)
    
    for feat_file in other_feats_list:
        try:
            feat_df = pandas.read_csv(feat_file)
            feat_df['Date'] = pandas.to_datetime(feat_df['Date'],infer_datetime_format =True)
            feat_df.index = feat_df['Date']
            feat_df = feat_df.sort_values(by='Date', ascending=True)
            del feat_df['Date']
            my_df = my_df.join(feat_df)
        except:
            pass
    
    return my_df.columns [2:]

#def loadSeriesToDataframe (path, filename, column_names=None):
#    #my_df = pandas.read_csv(path+'/'+filename, converters={'Change':p2f})
#    if column_names is not None:
#        #my_df = pandas.read_csv(path+'/'+filename, names=['Date', 'Close', 'Open', 'High', 'Low', 'Change'], converters={'Change':p2f})
#        my_df = pandas.read_csv(path+'/'+filename, names=column_names, converters={'Change':p2f})
#    else:
#        my_df = pandas.read_csv(path+'/'+filename, converters={'Change':p2f})
#        
#    my_df['Date'] = pandas.to_datetime(my_df['Date'],infer_datetime_format =True)
#    my_df.index = my_df['Date']
#    my_df = my_df.sort_values(by='Date', ascending=True)
#    del my_df['Date']
#
#    return my_df

def indexDf (df, index='Date'):
    df[index] = pd.to_datetime(df[index],infer_datetime_format =True)
    df.index = df[index]
    df = df.sort_values(by=index, ascending=True)
    del df[index]

    return df

def loadSeriesToDataframe (path, filename, column_names=None, 
                           other_feats_list=['./datasets/Macro/vix2.csv']):
    
    if column_names is not None:        
        my_df = pandas.read_csv(path+'/'+filename, names=column_names, converters={'Change':p2f})
    else:
        my_df = pandas.read_csv(path+'/'+filename, converters={'Change':p2f})
    my_df = indexDf (my_df, 'Date')
#==============================================================================
#     my_df['Date'] = pandas.to_datetime(my_df['Date'],infer_datetime_format =True)
#     my_df.index = my_df['Date']
#     my_df = my_df.sort_values(by='Date', ascending=True)
#     del my_df['Date']
#==============================================================================

    for feat_file in other_feats_list:
        try:
            feat_df = pandas.read_csv(feat_file)
            feat_df['Date'] = pandas.to_datetime(feat_df['Date'],infer_datetime_format =True)
            feat_df.index = feat_df['Date']
            feat_df = feat_df.sort_values(by='Date', ascending=True)
            del feat_df['Date']
            my_df = my_df.join(feat_df)
        except:
            pass
    return my_df


def relabelDataset (X, period_ahead=None, bCenter_y=False, return_type='log'):
    if period_ahead == None:
        period_ahead = np.int(np.random.uniform(low=1, high=22))
        
    y = np.zeros ((len(X),1))
    
    for i in range (len(y) - period_ahead):
        #y[i, 0] = (X[i+period_ahead,-1,0] / X[i,-1,0] - 1)
        #y[i, 0] = ((X[i+period_ahead,-1,0] / X[i,-1,0] - 1) / np.std(X[i,-22:,5]))
        chg = X[i,-(np.minimum(22,np.shape(X)[1]-1)):,0] / X[i,-(np.minimum(23, np.shape(X)[1])):-1,0] - 1
        sigma = np.std(chg) * 100

        if return_type=='linear':
            y[i, 0] = (X[i+period_ahead,-1,0] / X[i,-1,0] - 1) #/ sigma   #/(0.08/16)) #* ((252/period_ahead)**0.5)
        elif return_type=='log':
            y[i, 0] = np.log(X[i+period_ahead,-1,0] / X[i,-1,0]) #/ sigma
    
    if bCenter_y == True:
        y -= np.mean (y)
    print ("sigma: "+str(sigma))
    print ('y - mean: '+str(np.mean (y))+' , std: '+str(np.std(y)))
    return (y)
    
def rebuildVWithVariableLength (input_X,min_length=90):
    max_length = np.shape(input_X)[1]

    X = np.zeros (np.shape(input_X))
    
    for i in range (len(X)):
        pad_length = max_length - np.int(np.random.uniform(low=min_length, high=max_length))
        if pad_length > 0:
            X[i,0:pad_length,:] = 0
            X[i,pad_length:,:] = input_X[i,pad_length:,:]

    return X

def normalizeOnTheFly (X, mu_lookback=50, sigma_lookback=22,
                       mu_sigma_list =[0,1,2,3,5,6,7,8,23,24,25,34,36,46,47,48,49,50,51,56,58,59,72,79],
                        by100_list=[16,17,19,20,21,27,28,29,30,31,52,53,54,55,60,61,62,65,66,68,71,73,74,75,76,78,80],
                            volume_feat_list=[]):
    x_s = np.shape(X)
    
    
    try:
        for i in range (x_s[0]):
            
            mu = np.mean (X[i,-mu_lookback:,0])
            
            
            sigma = np.std (X[i,-sigma_lookback:,0]) #/ sigma
            
            for j in mu_sigma_list:
                X[i,:,j] = (X[i,:,j] - mu) #/ sigma
    #        X[i,:,1] = (X[i,:,1]  - mu) / sigma
    #        X[i,:,2] = (X[i,:,2]  - mu) / sigma
    #        X[i,:,3] = (X[i,:,3]  - mu) / sigma
    #        X[i,:,5] = (X[i,:,5]  - mu) / sigma
    #        X[i,:,6] = (X[i,:,6]  - mu) / sigma
    #        X[i,:,7] = (X[i,:,7]  - mu) / sigma
    #        X[i,:,8] = (X[i,:,8]  - mu) / sigma
    #        X[i,:,23] = (X[i,:,23]  - mu) / sigma
    #        X[i,:,24] = (X[i,:,24]  - mu) / sigma
    #        X[i,:,25] = (X[i,:,25]  - mu) / sigma
    #        X[i,:,34] = (X[i,:,34]  - mu) / sigma
    #        X[i,:,36] = (X[i,:,36]  - mu) / sigma
    #        X[i,:,46] = (X[i,:,46]  - mu) / sigma
    #        X[i,:,47] = (X[i,:,47]  - mu) / sigma
    #        X[i,:,48] = (X[i,:,48]  - mu) / sigma
    #        X[i,:,49] = (X[i,:,49]  - mu) / sigma
    #        X[i,:,50] = (X[i,:,50]  - mu) / sigma
    #        X[i,:,51] = (X[i,:,51]  - mu) / sigma
    #        X[i,:,56] = (X[i,:,56]  - mu) / sigma
    #        X[i,:,58] = (X[i,:,58]  - mu) / sigma
    #        X[i,:,59] = (X[i,:,69]  - mu) / sigma
    #        X[i,:,72] = (X[i,:,72]  - mu) / sigma
    #        X[i,:,79] = (X[i,:,79]  - mu) / sigma
            
            for j in by100_list:
                X[i,:,j] /= 100
    #        X[i,:,16] /= 100
    #        X[i,:,17] /= 100
    #        X[i,:,19] /= 100
    #        X[i,:,20] /= 100
    #        X[i,:,21] /= 100
    #        X[i,:,27] /= 100
    #        X[i,:,28] /= 100
    #        X[i,:,29] /= 100
    #        X[i,:,30] /= 100
    #        X[i,:,31] /= 100
    #        X[i,:,52] /= 100
    #        X[i,:,53] /= 100
    #        X[i,:,54] /= 100
    #        X[i,:,55] /= 100
    #        X[i,:,60] /= 100
    #        X[i,:,61] /= 100
    #        X[i,:,62] /= 100
    #        X[i,:,65] /= 100
    #        X[i,:,66] /= 100
    #        X[i,:,68] /= 100
    #        X[i,:,71] /= 100
    #        X[i,:,73] /= 100
    #        X[i,:,74] /= 100
    #        X[i,:,75] /= 100
    #        X[i,:,76] /= 100
    #        X[i,:,78] /= 100
    #        X[i,:,80] /= 100        
            #X[i,:,4] = X[i,:,4] / np.mean (X[i,:,4])
            for j in volume_feat_list:
                X[i,:,j] /= np.mean (X[i,-mu_lookback:,j])
    except:
        print ('i: '+str(i))
                
    return (X)

#this is a key functions that combines two dataframes
#the key feature is that values of the low frequency dataframe df2
#are repeated for every value of df1 until a new value of df2 arrives
def combine_dataframes (df1, df2, delay = 0):    
    if type(df2) == pd.core.series.Series:
        df2 = df2.to_frame (name = df2.name)
        
    idx_fast_str = df1.index
    idx_slow_str = df2.index
    
    for col in df2.columns:
        if col in df1.columns:
            del df1[col]
            
    cols1 = [col for col in df1.columns]
    cols2 = [col for col in df2.columns]
    
    fast_to_slow_indexing = np.zeros (len(idx_fast_str), int)
    i = 0
    j = 0
    while idx_slow_str[j] >= idx_fast_str[i]:
        i+=1
    
    for j in range (len(idx_slow_str)-1):
        try:
            while (idx_fast_str[i] >= idx_slow_str[j]) and \
             (idx_fast_str[i] < idx_slow_str[j+1]) and \
                (i < len (idx_fast_str)):
                    fast_to_slow_indexing [i] = j
                    i += 1
        except:
            break
    
    fast_to_slow_indexing -= delay
    offset = np.count_nonzero(fast_to_slow_indexing<0)
    
    new_data = np.hstack((np.array(df1[offset:]), 
                          (np.array (df2)) [np.minimum(fast_to_slow_indexing [offset:], len (df2) - 1)]))
    
    return pd.DataFrame (data = new_data,
                              index = idx_fast_str[offset:],
                              columns = cols1 + cols2)
    

def get_from_time(last_timestamp, tf, days = None):
    dti = pd.to_datetime([last_timestamp])
    ts = datetime.datetime(dti.year, dti.month, dti.day,
            dti.hour, dti.minute, dti.second)
    if tf == 'D':
        from_time = str(ts - relativedelta(days=(900 if days is None else days) ))
    elif tf == 'H4':
        from_time = str(ts - relativedelta(days=(200 if days is None else days) ))
    elif tf == 'H1':
        from_time = str(ts - relativedelta(days=(90 if days is None else days) ))
    elif tf == 'M15':
        from_time = str(ts - relativedelta(days=(40 if days is None else days) ))
    return from_time