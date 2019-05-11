#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:15:10 2017

@author: renato
"""

import pandas as pd
import numpy as np

filename = 'USD_bull_market_without_ZAR_gap_0.csv'
xv_file = '/home/renato/Desktop/Projects/Trading/Trading/Training/Candle_Strategies/Analysis/Candles_mix03_xv1.csv'

full_df = pd.read_csv(filename)
st_raw = np.zeros (len(full_df), str)

for key in list(full_df.columns[6:]):
    st_raw += full_df[key].astype(str)

full_df['strategy_raw'] = st_raw
#full_df['Net_wins'] = (2*full_df.acc - 1.0) * full_df.No_preds
    
strats = list(set(full_df.strategy_raw))
hit_ratio_arr = np.zeros (len (strats))
neg_ratio_arr = np.zeros (len (strats))
concentration_arr = np.zeros (len (strats))
wins_arr = np.zeros (len (strats))

winning_strats = []

for i, strat in enumerate(strats):
    sub_df = full_df[full_df.strategy_raw == strat]

    net_wins = (2*sub_df.acc.astype(float) - 1.0) * sub_df.No_preds.astype(float)
    tot_wins = np.sum (sub_df.acc.astype(float) * sub_df.No_preds.astype(float))
    avg_wins = np.nanmean (net_wins)
    
    tot_preds = np.sum(sub_df.No_preds.astype(float))
    hit_ratio = tot_wins / tot_preds
    
    neg = np.sum(net_wins[sub_df.acc < 0.5])
    neg_over_total = -neg / tot_wins
    try:
        concentration = np.sum(net_wins[sub_df.acc > 0.5]**2) / (np.sum(net_wins[sub_df.acc > 0.5])**2)
    except:
        concentration = 100
    
    hit_ratio_arr[i] = hit_ratio
    neg_ratio_arr [i] = neg_over_total
    concentration_arr [i] = concentration
    wins_arr [i] = np.sum(net_wins)
    
    
    
    if hit_ratio > 0.65 and np.nanmedian(sub_df.p_val.astype(float)) < 0.05:# and np.nanmedian(sub_df.p_val[(sub_df.acc.astype(float)>0.5)].astype(float)) < 0.15 and np.nanmedian(sub_df.p_val[(sub_df.acc.astype(float)<0.5)].astype(float)) > 0.35:
    #and np.nanmean(sub_df.p_val[(sub_df.acc>0.5)]) < 0.1 and np.nanmean(sub_df.p_val[(sub_df.acc<0.5)]) > 0.25 :
        print (str(i) + ' - hit ratio: ' + str(hit_ratio) + ', p_val: ' + str(np.nanmedian(sub_df.p_val.astype(float))))
        winning_strats.append (strat)
       
if True:        
    xv_df = pd.read_csv (xv_file)
    st_raw = np.zeros (len(xv_df), str)
    
    for key in list(xv_df.columns[6:]):
        st_raw += xv_df[key].astype(str)
    
    xv_df['strategy_raw'] = st_raw

    #winning_strats = strats

    xv_hit_ratio_arr = np.zeros (len (winning_strats))
    xv_neg_ratio_arr = np.zeros (len (winning_strats))
    xv_concentration_arr = np.zeros (len (winning_strats))
    xv_preds_arr = np.zeros (len (winning_strats))
    xv_wins_arr = np.zeros (len (winning_strats))
    xv_wins_list = []
    
    for i, strat in enumerate(winning_strats):
        sub_df = xv_df[xv_df.strategy_raw == strat]
    
        net_wins = (2*sub_df.acc - 1.0) * sub_df.No_preds
        tot_wins = np.sum (sub_df.acc * sub_df.No_preds)
        avg_wins = np.nanmean (net_wins)    
        tot_preds = np.sum(sub_df.No_preds)
        neg = np.sum(net_wins[sub_df.acc < 0.5])
        
        try:
            #print (str (i) +'th winning strategy, df shape: ' + str (np.shape(sub_df)))
        
            print ('Total wins: ' + str(tot_wins) + ', no_pred: ' + str(tot_preds) + ', acc: ' + str (tot_wins/tot_preds))       
            neg_over_total = 0.0
            concentration = 0.0
            #neg_over_total = -neg / tot_wins
            hit_ratio = tot_wins / tot_preds
            #concentration = np.sum(net_wins[sub_df.acc > 0.5]**2) / (np.sum(net_wins[sub_df.acc > 0.5])**2)
        except:
            concentration = np.nan
            hit_ratio = np.nan
            neg_over_total = np.nan
        
        xv_hit_ratio_arr[i] = hit_ratio
        xv_neg_ratio_arr [i] = neg_over_total
        xv_concentration_arr [i] = concentration
        xv_preds_arr [i] = tot_preds
        xv_wins_arr [i] = np.sum (net_wins)
        xv_wins_list.append (net_wins)
        
    print ('Summary: ' + str(np.sum(xv_preds_arr)) + 'predictions, net wins: ' + str(np.sum(xv_wins_arr)))
    print ('Net: ' + str(np.sum(xv_wins_arr)/np.sum(xv_preds_arr)))
    print ('Average hit ratio: ' + str(np.nanmean(xv_hit_ratio_arr)))
    print ('No strategies: ' + str(len(winning_strats)))

tot_preds = 0
for i, strat in enumerate(winning_strats):
    sub_df = full_df[full_df.strategy_raw == strat]
    tot_preds += np.sum(sub_df.No_preds)
    
    
weights = np.zeros (len (winning_strats))
for i, strat in enumerate(winning_strats):
    weights [i] = float (tot_preds) / float(np.sum(full_df[full_df.strategy_raw == strat].No_preds))
    

weights /= np.sum(weights)

weights = np.minimum (weights, 0.2)

weights /= np.sum(weights)


cum = 0
cum_w = 0
a = (weights * xv_hit_ratio_arr)
for u, w in zip(a, weights):
    if np.isnan(u):
        pass
    else:
        cum += u
        cum_w += w
        
print (float(cum)/float(cum_w))