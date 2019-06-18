import numpy as np
from Framework.Dataset.Dataset import *

def new_high_over_lookback_window (ds, args={}):
    if 'lookback_window' in args:
        lookback_window = args['lookback_window']
    else:
        lookback_window = 252
        
    if 'feat' in args:
        feat = args['feat']
    else:
        feat = 'Close'
        
    if 'step_width' in args:
        step_width = args['step_width'] #used to convolve
    else:
        step_width = 20
        
    print ('Step width: ' + str (step_width))
        
    new_feat = np.zeros ((np.shape(ds.X)[0], np.shape(ds.X)[1], 1))
    idx_feat = ds.getFeatIdx(feat)
    
    for i in range (lookback_window, ds.X.shape[0]-1):
        new_feat [i, -1, 0] = 0

        if ds.X[i, -1, idx_feat] == np.max (ds.X[i-lookback_window:i+1, -1, idx_feat]):
            new_feat [i, -1, 0] = 1.0
        elif ds.X[i, -1, idx_feat] == np.min (ds.X[i-lookback_window:i+1, -1, idx_feat]):
            new_feat [i, -1, 0] = -1.0

    new_feat [:, 0, 0] =  np.minimum(np.maximum(np.convolve(new_feat[:,0,0], np.ones(step_width))[:ds.X.shape[0]], -1.0), 1.0)    
    
    return new_feat
    

def crossover (ds, args={}):    

    if 'step_width' in args:
        step_width = args['step_width']
    else:
        step_width = 60
        
    if 'crossover_threshold' in args:
        crossover_threshold = args['crossover_threshold']
    else:
        crossover_threshold = 0.000000002
    
    if 'fast' in args:
        fast = args['fast']
    else:
        fast = 'ma_50_close'
    
    if 'slow' in args:
        slow = args['slow']
    else:
        slow = 'ma_200_close'
    
    if 'metric' in args:
        metric = args['metric']
    else:
        metric = 'crossover_window'
        
    print ('fast: '+ fast + ', slow: '+ slow)
        
    new_feat = np.zeros ((np.shape(ds.X)[0], np.shape(ds.X)[1], 1))
    idx_fast = ds.getFeatIdx(fast)
    idx_slow = ds.getFeatIdx(slow)
    
    for i in range (ds.X.shape[0]):
        if metric == 'crossover_window':
            new_feat [i, -1, 0] = 0
            if ds.X[i-1, -1, idx_fast] / ds.X[i-1, -1, idx_slow] < (1-crossover_threshold) and\
                    ds.X[i, -1, idx_fast] / ds.X[i, -1, idx_slow] > (1+crossover_threshold):
                    new_feat [i, -1, 0] = 1
            elif ds.X[i-1, -1, idx_fast] / ds.X[i-1, -1, idx_slow] > (1+crossover_threshold)  and\
                    ds.X[i, -1, idx_fast] / ds.X[i, -1, idx_slow] < (1-crossover_threshold) :
                    new_feat [i, -1, 0] = -1
    new_feat [:, 0, 0] =  np.minimum(np.maximum(np.convolve(new_feat[:,0,0], np.ones(step_width))[:ds.X.shape[0]], -1.0), 1.0)    
    
    return new_feat

def feat_metrics (ds, args={}):    

    if 'lookback_window' in args:
        lookback_window = args['lookback_window']
    else:
        lookback_window = 60
    
    if 'feat' in args:
        feat = args['feat']
    else:
        feat = 'RSI'
    
    if 'metric' in args:
        metric = args['metric']
    else:
        metric = 'peak'
        
    new_feat = np.zeros ((np.shape(ds.X)[0], np.shape(ds.X)[1], 1))
    idx = ds.getFeatIdx(feat)
    
    for i in range (lookback_window, ds.X.shape[0]):
        if metric == 'peak':
            new_feat [i, -1, 0] = np.max(ds.X[i-lookback_window:i,-1,idx])
        elif metric == 'bottom':
            new_feat [i, -1, 0] = np.min(ds.X[i-lookback_window:i,-1,idx])
        elif metric == 'close_minus_peak':
            new_feat [i, -1, 0] = ds.X[i,-1,idx] - np.max(ds.X[i-lookback_window:i,-1,idx])
        elif metric == 'close_minus_bottom':
            new_feat [i, -1, 0] = ds.X[i,-1,idx] - np.min(ds.X[i-lookback_window:i,-1,idx])
        elif metric == 'close_minus_average':
            new_feat [i, -1, 0] = ds.X[i,-1,idx] - np.mean(ds.X[i-lookback_window:i,-1,idx])
        elif metric == 'close_minus_average_in_sigmas':
            new_feat [i, -1, 0] = (ds.X[i,-1,idx] - np.mean(ds.X[i-lookback_window:i,-1,idx])) / np.std(ds.X[i-lookback_window:i,-1,idx]) 
    
    return new_feat