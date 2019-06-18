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

from Framework.Dataset.Dataset import * 

np.random.seed(50)

def getHighLows (x):
    y = sorted(list(x.items())[:], key=operator.itemgetter(1))
    y2 = sorted(list(x.items())[:], key=operator.itemgetter(1), reverse=True)
    arr = np.array(y)
    arr2 = np.array(y2)
    a = arr[:,0]
    a2 = arr2[:,0]
    dict_l = {}
    dict_h = {}
    #print ( str(a[0])+ ': ' +str(len(a)))
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
        self.mu, self.sigma = getMuSigma (list(x.values()))
        self.bUpLine = bUpLine
        self.lows = None
        self.highs = None
        
        self.threshold_no_days = threshold
        self.threshold_high_low = threshold_hl
        
        self.exclude_last_pts = 25  #not considered for new highs or new lows, to avoid trendlines btw a new low or new high and a point in the past
        
        #self.no_lines = 0
        self.upward_lines_list = []        #contains all possible lines upward lines
        self.upward_standing_lines_list = [] #contains only the lines that have not yet been broken
        self.upward_relevant_lines_list = [] #contains only meaningful trendlines - TBD
        self.upward_recent_lines_list = [] #contains recent trendlines only        
        self.upward_lines_status_list = [] #has a boolean corresponding to each line in the list above
        
        self.downward_lines_list = []        #contains all possible lines downward lines
        self.downward_standing_lines_list = [] #contains only the lines that have not yet been broken
        self.downward_relevant_lines_list = [] #contains only meaningful trendlines - TBD
        self.downward_recent_lines_list = [] #contains recent trendlines only        
        self.downward_lines_status_list = [] #has a boolean corresponding to each line in the list above       

        self.standing_lows_list = []
        self.standing_highs_list = []
        self.relevant_lows_list = []
        self.relevant_highs_list = []
        self.recent_lows_list = []
        self.recent_highs_list = []        
        
        #init highs and lows dicts
        self.lows, self.highs = dict_pts, dummy = getHighLows(x) #these are dictionaires containing all
                                                                    #candidates highs and lows
        #calc's all candidate lines
        self.generateLinesLists ()
        
                
    def generateLinesLists (self):
        #first the upward lines
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  v1 >= self.threshold_no_days and k1 <= len(self.time_series) - self.exclude_last_pts)
        for k1, v1 in d1.items ():
            
            d2 = dict((k2, v2) for k2, v2 in d1.items() if 
                          (k2 > k1 and list(self.time_series.values())[np.int(k2)] >= list(self.time_series.values())[np.int(k1)]))
            
            for k2, v2 in d2.items ():                   
                self.upward_lines_list.append ([[k1,k2],[list(self.time_series.values())[np.int(k1)], list(self.time_series.values())[np.int(k2)]]])
                self.upward_lines_status_list.append(True)                
        
        #then, the downward ones
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  v1 >= self.threshold_no_days and k1 <= len(self.time_series) - self.exclude_last_pts)
        for k1, v1 in d1.items ():
            
            d2 = dict((k2, v2) for k2, v2 in d1.items() if 
                          (k2 > k1 and list(self.time_series.values())[np.int(k2)] <= list(self.time_series.values())[np.int(k1)]))
            for k2, v2 in d2.items ():
                self.downward_lines_list.append ([[k1,k2],[list(self.time_series.values())[np.int(k1)], list(self.time_series.values())[np.int(k2)]]])
                self.downward_lines_status_list.append(True)
                #self.no_lines += 1        
    
    def checkStandingHighsLows (self):
        #first the lows
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  v1 >= self.threshold_high_low * len(self.time_series))
        
        self.standing_lows_list = []
        
        for k1, v1 in d1.items ():
            a = list(self.time_series.values ())[np.int(k1):]
            b = a[0]
            if (np.min(a-b)/self.mu < -1.5 * self.sigma):
                pass

            else:
                self.standing_lows_list.append ([k1,b])
        
        #then highs
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  v1 >= self.threshold_high_low * len(self.time_series))
        
        self.standing_highs_list = []
        
        for k1, v1 in d1.items ():
            a = list(self.time_series.values ())[np.int(k1):]
            b = a[0]

            if (np.min(b-a)/self.mu < -1.5 * self.sigma):
                #print ('Low broken')
                pass
            else:
                self.standing_highs_list.append ([k1,v1])
                
    def removeInvalidLines (self):
        self.checkValidLines (bUp=True)
        self.checkValidLines (bUp=False)        
    
    #modifies attribute self.lines_status_list
    #if line was broken between two extremes, the entry for this line in the status list will be False, otherwise, it remais True
    #this method is to get rid of excessive number of invalid trendlines
    def checkValidLines (self, bUp=True):
        lines_list, lines_status_list = self.selectLines (bUp=bUp)
        
        for i in range (len(lines_status_list)):
            lines_status_list [i] = True

        #only_valid_lines_list = []

        counter = 0
        for [k1,k2],[v1,v2] in lines_list:            
            a = list(self.time_series.values ())[np.int(k1):np.int(k2)+1]
            b = ((np.linspace(k1,k2,k2-k1+1).astype(float)) - k1) * (v2 - v1) / (k2-k1) + v1
            
            aux1 = lines_list.pop (counter)
            aux2 = lines_status_list.pop (counter)

            if (bUp == False and (np.min(b-a)/self.mu < -1.5 * self.sigma)) or (bUp== True and (np.min(a-b)/self.mu < -1.5 * self.sigma)):            
                #lines_status_list[counter] = False
                pass
            else:
                lines_list.insert(counter, aux1)
                lines_status_list.insert(counter, aux2)
                #only_valid_lines_list.append ([[k1,k2],[v1,v2]])
            
            counter += 1        
    
    def checkStandingLines (self, bUp=True):
        lines_list, lines_status_list = self.selectLines (bUp=bUp)
        
        lines_standing_list = []
        counter = 0
        for [k1,k2],[v1,v2] in lines_list:
            if lines_status_list [counter] == True:
                a = list(self.time_series.values ())[np.int(k1):]
                
                b = ((np.linspace(k1,len(a)+k1-1,len(a)).astype(float)) - k1) * (v2 - v1) / (k2-k1) + v1
                
                if (bUp == False and (np.min(b-a)/self.mu < -1.5 * self.sigma) or (bUp==True and (np.min(a-b)/self.mu < -1.5 * self.sigma))):                        
                    lines_status_list[counter] = False
                else:
                    lines_standing_list.append ([[k1,k2],[v1,v2]])
            counter += 1
    
    #selects either upward or downward lines
    def selectLines (self, bUp=True):
        if bUp == True:
            lines_list = self.upward_lines_list
            lines_status_list = self.upward_lines_status_list
        else:
            lines_list = self.downward_lines_list
            lines_status_list = self.downward_lines_status_list
        return lines_list, lines_status_list
    
    def plotValidTrendlines (self, bUp=True):            
        lines_list, lines_status_list = self.selectLines (bUp=bUp)        
            
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        for i, status in enumerate(lines_status_list):
            if status == True:
                [k1, k2],[v1,v2] = lines_list[i]
                plt.plot ([k1, k2],[v1,v2])
        plt.show ()
    
    def plotStandingTrendlines (self, bUp=True):
        if self.upward_lines_status_list.count(False) == 0 and self.downward_lines_status_list.count(False) == 0:
            self.checkValidLines(bUp=bUp)
            self.checkStandingLines(bUp=bUp)
        
        lines_list, lines_status_list = self.selectLines (bUp=bUp)        
        
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        for i, status in enumerate(lines_status_list):
            if status == True:
                [k1, k2],[v1,v2] = lines_list[i]
                a = list(self.time_series.values ())[np.int(k1):]
                plt.plot ([k1, len(self.time_series)-1],[v1,(len(self.time_series)-1-k1)*(v2-v1)/(k2-k1)+v1])
        plt.show ()        

    def plotStandingHighLows (self):
        if len (self.standing_lows_list) == 0 and len (self.standing_highs_list) == 0:
            self.checkStandingHighsLows ()
        
        fig = plt.figure ()
        plt.plot(self.time_series.keys(), self.time_series.values())
        
        for k1, v1 in self.standing_lows_list:
            b = list(self.time_series.values ())[np.int(k1)]
            plt.plot ([k1, len(self.time_series)-1],[b,b], c='red')
        
        for k1, v1 in self.standing_highs_list:
            b = list(self.time_series.values ())[np.int(k1)]
            plt.plot ([k1, len(self.time_series)-1],[b,b], c='blue')
            
        plt.show ()
        
    def identifyRelevantHighLows (self, relevant_threshold=50, exclude_last_pts=None, plot=False):
        if exclude_last_pts == None:
            exclude_last_pts = relevant_threshold
        
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  v1 >= relevant_threshold and k1 <= len(self.time_series) - exclude_last_pts)        
        
        self.relevant_lows_list = []
        if plot == True:
            fig = plt.figure ()
            plt.plot(self.time_series.keys(), self.time_series.values())
        
        for k1, v1 in d1.items ():
            b = list(self.time_series.values ())[np.int(k1)]
            self.relevant_lows_list.append ([k1,v1])
            if plot == True:
                plt.plot ([k1, len(self.time_series)-1],[b,b])
            
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  v1 >= relevant_threshold and k1 <= len(self.time_series) - exclude_last_pts)
        
        self.relevant_highs_list = []
        for k1, v1 in d1.items ():
            b = list(self.time_series.values ())[np.int(k1)]
            self.relevant_highs_list.append ([k1,v1])
            if plot == True:
                plt.plot ([k1, len(self.time_series)-1],[b,b])
        if plot == True:
            plt.show ()
            
    def calcDistanceSpotTrendOrHighLow (self, trend):
        spot = list(self.time_series.values())[-1]

        if np.shape (trend) == (2,2):
            #this is a trendline
            [k1, k2],[v1,v2] = trend
            
            return (spot - ((len(self.time_series)-1-k1)*(v2-v1)/(k2-k1)+v1))
        elif np.shape (trend) == (2,):
            #this is a either a high or a low
            return (spot - list(self.time_series.values ())[np.int(trend[0])])
        
    def identifyRecentHighLows (self, recent_threshold=150, threshold=10, plot=False):
        d1 = dict((k1, v1) for k1, v1 in self.lows.items() if 
                  len(self.time_series.values()) - k1 <= recent_threshold and v1 >= threshold)
        if plot == True:
            fig = plt.figure ()
            plt.plot(self.time_series.keys(), self.time_series.values())
        
        self.recent_lows_list = []
        for k1, v1 in d1.items ():
            self.recent_lows_list.append ([k1,v1])            
            if plot == True:
                b = list(self.time_series.values ())[np.int(k1)]
                plt.plot ([k1, len(self.time_series)-1],[b,b], c='red')
            
        d1 = dict((k1, v1) for k1, v1 in self.highs.items() if 
                  len(self.time_series.values()) - k1 <= recent_threshold and v1 >= threshold)
        
        self.recent_highs_list = []
        for k1, v1 in d1.items ():
            self.recent_highs_list.append ([k1,v1])
            if plot == True:
                b = list(self.time_series.values ())[np.int(k1)]
                plt.plot ([k1, len(self.time_series)-1],[b,b], c='blue')
        if plot == True:
            plt.show ()       
