# -*- coding: utf-8 -*-
try:
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, WeekdayLocator,\
        DayLocator, MONDAY
    from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
except:
    pass
import numpy as np
import pandas


def compute_return (model, X, y, threshold=0.4):
    print ('shape X: '+str(X.shape))
    #try:
    if True:
        pred_prob = model.predict(X, batch_size=32)
        pred = np.zeros (len(pred_prob))
        for i in range(len (pred_prob)):
            if pred_prob[i,0] > threshold and pred_prob[i,0] > pred_prob[i,2] and pred_prob[i,0] > pred_prob[i,1]:
                pred[i] = 0
            elif pred_prob[i,2] > threshold and pred_prob[i,2] > pred_prob[i,0] and pred_prob[i,2] > pred_prob[i,1]:
                pred[i] = 2
            else:
                pred[i] = 1
    
        pos_pred = pred[pred!=1]
        pos_y = y[pred!=1]
    
        ret = 0.0
        long_hits = 0
        long_misses = 0
        short_hits = 0
        short_misses = 0
        for j in range(len (pos_pred)):
            if pos_pred [j]== 0:
                if pos_y[j,0] == True:
                    ret += 5
                    short_hits += 1
                else:
                    ret += -3
                    short_misses += 1
            else:
                if pos_y[j,2] == True:
                    ret += 5
                    long_hits += 1
                else:
                    ret += -3
                    long_misses += 1
        return ret, long_hits, long_misses, short_hits, short_misses