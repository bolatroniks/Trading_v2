#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:33:34 2017

@author: renato
"""

import numpy as np
import copy
import operator
from matplotlib import pyplot as plt

from Trading.Dataset.Dataset import * 

np.random.seed(50)

def getHighLows (x):
    y = sorted(x.items()[:-25], key=operator.itemgetter(1))
    y2 = sorted(x.items()[:-25], key=operator.itemgetter(1), reverse=True)
    arr = np.array(y)
    arr2 = np.array(y2)
    a = arr[:,0]
    a2 = arr2[:,0]
    dict_l = {}
    dict_h = {}
    print ( str(a[0])+ ': ' +str(len(a)))
    dict_l[a[0]] = len(a)
    dict_h[a2[0]] = len(a2)
    for i in range (1,len(arr)):
        dict_l[a[i]] = np.min(np.abs(a[i] - a[0:i]))
        dict_h[a2[i]] = np.min(np.abs(a2[i] - a2[0:i]))
    return dict_l, dict_h
    
def getMuSigma (x_t, last=100):
    mu = np.mean(x_t[-last:])
    sigma = np.std(np.diff(x_t)[-last:]) / mu
    return mu, sigma

def genPath (n):
    inc = np.random.randn(n)
    X = np.cumprod(1.0+inc/100)
    my_dict = {}
    for i, elem in enumerate(X):
        my_dict[i] = elem
    return my_dict

class TrendlineSet:
    def __init__ (self, x, bUpLine=True, threshold=15, threshold_hl=0.025):
        if type(x) != dict:
            if type(x) == np.ndarray:
                my_dict = {}
                for i, elem in enumerate(x):
                    my_dict[i] = elem
                x = my_dict
        
        self.time_series = x
        self.mu, self.sigma = getMuSigma (x.values())
        self.bUpLine = bUpLine        
        if self.bUpLine == True:
            self.lows, self.highs = dict_pts, dummy = getHighLows(x)
        else:
            self.lows, self.highs = dummy, dict_pts = getHighLows(x)
        
        self.threshold_no_days = threshold
        self.threshold_high_low = threshold_hl
        
        d1 = dict((k1, v1) for k1, v1 in dict_pts.items() if 
                  v1 >= self.threshold_no_days)
        
        self.no_lines = 0
        self.lines_list = []        #contains all possible lines connecting highs or lows
        self.standing_lines_list = [] #contains only the lines that have not yet been broken
        self.relevant_lines_list = [] #contains only meaningful trendlines
        self.recent_lines_list = [] #contains recent trendlines only
        
        self.lines_status_list = [] #has a boolean corresponding to each line in the list above       

        self.standing_lows_list = []
        self.standing_highs_list = []
        self.relevant_lows_list = []
        self.relevant_highs_list = []
        self.recent_lows_list = []
        self.recent_highs_list = []    
        
        for k1, v1 in d1.items ():
            if self.bUpLine == True:
                d2 = dict((k2, v2) for k2, v2 in d1.items() if 
                          (k2 > k1 and x.values()[np.int(k2)] >= x.values()[np.int(k1)]))
            else:
                d2 = dict((k2, v2) for k2, v2 in d1.items() if 
                          (k2 > k1 and x.values()[np.int(k2)] <= x.values()[np.int(k1)]))
            for k2, v2 in d2.items ():   
                #plt.plot([k1,k2],[x.values()[np.int(k1)], x.values()[np.int(k2)]])
                self.lines_list.append ([[k1,k2],[x.values()[np.int(k1)], x.values()[np.int(k2)]]])
                self.lines_status_list.append(True)
                self.no_lines += 1
    
    def checkStandingHighsLows (self):
        #first the lows
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  v1 >= self.threshold_high_low * len(self.time_series))
        
        self.standing_lows_list = []
        
        for k1, v1 in d1.items ():
            a = self.time_series.values ()[np.int(k1):]
            b = a[0]

            #fig = plt.figure ()
            #plt.plot(np.linspace(0,len(a)-1, len(a)), a)
            #plt.plot(np.linspace(0,len(a)-1, len(a)), b*np.ones(len(a)))
            #plt.show ()

            #print ('Low: '+str(k1)+', '+str (v1))
            if (np.min(a-b)/self.mu < -1.5 * self.sigma):
                pass
                #print ('Low broken')
            else:
                self.standing_lows_list.append ([k1,b])
        
        #then highs
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  v1 >= self.threshold_high_low * len(self.time_series))
        
        self.standing_highs_list = []
        
        for k1, v1 in d1.items ():
            a = self.time_series.values ()[np.int(k1):]
            b = a[0]

            #fig = plt.figure ()
            #plt.plot(np.linspace(0,len(a)-1, len(a)), a)
            #plt.plot(np.linspace(0,len(a)-1, len(a)), b*np.ones(len(a)))
            #plt.show ()

            #print ('Low: '+str(k1)+', '+str (v1))
            if (np.min(b-a)/self.mu < -1.5 * self.sigma):
                #print ('Low broken')
                pass
            else:
                self.standing_highs_list.append ([k1,v1])
                
            
    
    #modifies attribute self.lines_status_list
    #if line was broken between two extremes, the entry for this line in the status list will be False, otherwise, it remais True
    #this method is to get rid of excessive number of invalid trendlines
    def checkValidLines (self):
        for i in range (len(self.lines_status_list)):
            self.lines_status_list [i] = True
        counter = 0
        for [k1,k2],[v1,v2] in self.lines_list:
            #print k1
            a = self.time_series.values ()[np.int(k1):np.int(k2)+1]
            b = ((np.linspace(k1,k2,k2-k1+1).astype(float)) - k1) * (v2 - v1) / (k2-k1) + v1
            #fig = plt.figure ()
            #plt.plot(a)
            #plt.plot(b)
            #plt.show ()
            if self.bUpLine == False:
                #print ('min: '+str(np.min(b-a)))
                if (np.min(b-a)/self.mu < -1.5 * self.sigma):
                    #print ('Line broken')
                    self.lines_status_list[counter] = False
            else:
                if (np.min(a-b)/self.mu < -1.5 * self.sigma):
                    #print ('Line broken')
                    self.lines_status_list[counter] = False
                #print ('min: '+str(np.min(a-b)))
            counter += 1
    
    def checkStandingLines (self):
        counter = 0
        for [k1,k2],[v1,v2] in self.lines_list:
            if self.lines_status_list [counter] == True:
                a = self.time_series.values ()[np.int(k1):]
                
                b = ((np.linspace(k1,len(a)+k1-1,len(a)).astype(float)) - k1) * (v2 - v1) / (k2-k1) + v1
                #fig = plt.figure ()
                #plt.plot(a)
                #plt.plot(b)
                #plt.show ()
                if self.bUpLine == False:
                    #print ('min: '+str(np.min(b-a)))
                    if (np.min(b-a)/self.mu < -1.5 * self.sigma):
                        #print ('Line broken')
                        self.lines_status_list[counter] = False
                else:
                    if (np.min(a-b)/self.mu < -1.5 * self.sigma):
                        #print ('Line broken')
                        self.lines_status_list[counter] = False
                    #print ('min: '+str(np.min(a-b)))
            counter += 1
    
    def plotValidTrendlines (self):
        if self.lines_status_list.count(False) == 0:
            self.checkValidLines()
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        for i, status in enumerate(self.lines_status_list):
            if status == True:
                [k1, k2],[v1,v2] = self.lines_list[i]
                plt.plot ([k1, k2],[v1,v2])
        plt.show ()
    
    def plotStandingTrendlines (self):
        if self.lines_status_list.count(False) == 0:
            self.checkValidLines()
            self.checkStandingLines ()
            
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        for i, status in enumerate(self.lines_status_list):
            if status == True:
                [k1, k2],[v1,v2] = self.lines_list[i]
                a = self.time_series.values ()[np.int(k1):]
                plt.plot ([k1, len(self.time_series)-1],[v1,(len(self.time_series)-1-k1)*(v2-v1)/(k2-k1)+v1])
        plt.show ()        

    def plotStandingHighLows (self):
        if len (self.standing_lows_list) == 0 and len (self.standing_highs_list) == 0:
            self.checkStandingHighsLows ()
        
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        
        for k1, v1 in self.standing_lows_list:
            b = self.time_series.values ()[np.int(k1)]
            plt.plot ([k1, len(self.time_series)-1],[b,b], c='red')
        
        for k1, v1 in self.standing_highs_list:
            b = self.time_series.values ()[np.int(k1)]
            plt.plot ([k1, len(self.time_series)-1],[b,b], c='blue')
            
        plt.show ()
        
    def getRelevantHighLows (self, relevant_threshold=50, plot=False):
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  v1 >= relevant_threshold)
        
        if plot == True:
            fig = plt.figure ()
            plt.plot(self.time_series.keys(), self.time_series.values())
        
        for k1, v1 in d1.items ():
            b = self.time_series.values ()[np.int(k1)]
            if plot == True:
                plt.plot ([k1, len(self.time_series)-1],[b,b])
            
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  v1 >= relevant_threshold)
        
        for k1, v1 in d1.items ():
            b = self.time_series.values ()[np.int(k1)]
            if plot == True:
                plt.plot ([k1, len(self.time_series)-1],[b,b])
        if plot == True:
            plt.show ()
        
    def plotRecentHighLows (self, recent_threshold=150, threshold=10):
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  len(self.time_series.values()) - k1 <= recent_threshold and v1 >= threshold)
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        
        for k1, v1 in d1.items ():
            b = self.time_series.values ()[np.int(k1)]
            plt.plot ([k1, len(self.time_series)-1],[b,b], c='red')
            
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  len(self.time_series.values()) - k1 <= recent_threshold and v1 >= threshold)
        
        for k1, v1 in d1.items ():
            b = self.time_series.values ()[np.int(k1)]
            plt.plot ([k1, len(self.time_series)-1],[b,b], c='blue')
        plt.show ()
        
##x = genPath (10000)
#
#ds = Dataset(featpath=r'./datasets/Fx/Featured/NotNormalizedNoVolume/', lookback_window=2)
#ds.loadSeriesByNo(17, bRelabel=False, bNormalize=False, bConvolveCdl=False)
#
#for i in range(100,len (ds.test_X),1):
#    ts_down = TrendlineSet(ds.test_X[:i,-1,0], bUpLine=True, threshold=25)
#    #ts_up.plotValidTrendlines()
#    #ts_up.checkStandingLines ()
#    ts_down.getRelevantHighLows (plot=True)
#    ts_down.plotRecentHighLows ()
#    ts_down.plotStandingTrendlines()
#    ts_down.plotStandingHighLows ()