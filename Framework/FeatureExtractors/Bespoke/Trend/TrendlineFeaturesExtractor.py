# -*- coding: utf-8 -*-

from Trading.FeatureExtractors.Bespoke.Trend.TrendlineSet import *
import copy

class TrendlineFeaturesExtractor ():
    def __init__ (self, ts=None, x=None, threshold=15, threshold_hl=0.025):
        if ts is None:
            if x is None:
                return None
            else:
                self.ts = TrendlineSet (x, threshold=15, threshold_hl=0.025)
        else:
            self.ts = ts
            
    #self explanatory / this is a more generic function
    def getCloserTrendHighLow (self, list_obj, 
                               relevant_threshold=50, 
                               exclude_last_pts=None):
        if exclude_last_pts == None:
            exclude_last_pts = relevant_threshold
        
        #first gets what's the closest high
        dist_closest_high = -31999
        closest_high = None
        for trend in list_obj:
            dist = self.ts.calcDistanceSpotTrendOrHighLow (trend)
            if dist > dist_closest_high:
                dist_closest_high = dist
                closest_high = copy.deepcopy (trend)
                
        #then, gets what's the closest low
        dist_closest_low = 31999
        closest_low = None
        for trend in list_obj:
            dist = self.ts.calcDistanceSpotTrendOrHighLow (trend)
            if dist < dist_closest_low:
                dist_closest_low = dist
                closest_low = copy.deepcopy (trend)
        
        return closest_high, closest_low, dist_closest_high, dist_closest_low
    
    #self explanatory
    def getCloserRelevantHighLow (self, relevant_threshold=50, exclude_last_pts=None):
        if exclude_last_pts == None:
            exclude_last_pts = relevant_threshold
        
        self.ts.identifyRelevantHighLows(relevant_threshold=relevant_threshold, 
                                        exclude_last_pts=exclude_last_pts, plot=False)
                
        
        #first gets what's the closest high
        dist_closest_high = -31999
        closest_high = None
        for trend in self.ts.relevant_highs_list:
            dist = self.ts.calcDistanceSpotTrendOrHighLow (trend)
            if dist > dist_closest_high:
                dist_closest_high = dist
                closest_high = copy.deepcopy (trend)
                
        #then, gets what's the closest low
        dist_closest_low = 31999
        closest_low = None
        for trend in self.ts.relevant_lows_list:
            dist = self.ts.calcDistanceSpotTrendOrHighLow (trend)
            if dist < dist_closest_low:
                dist_closest_low = dist
                closest_low = copy.deepcopy (trend)
        
        return closest_high, closest_low, dist_closest_high, dist_closest_low
            
            
    #self explanatory
    def getCloserStandingHighLow (self):
        self.ts.checkStandingHighsLows ()
        
        #first gets what's the closest high
        dist_closest_high = -31999
        closest_high = None
        for trend in self.ts.standing_highs_list:
            dist = self.ts.calcDistanceSpotTrendOrHighLow (trend)
            if dist > dist_closest_high:
                dist_closest_high = dist
                closest_high = copy.deepcopy (trend)
                
        #then, gets what's the closest low
        dist_closest_low = 31999
        closest_low = None
        for trend in self.ts.standing_lows_list:
            dist = self.ts.calcDistanceSpotTrendOrHighLow (trend)
            if dist < dist_closest_low:
                dist_closest_low = dist
                closest_low = copy.deepcopy (trend)
        
        return closest_high, closest_low, dist_closest_high, dist_closest_low
    
    #compares each high and each low to the previous one
    def getHHHLScore (self):
        
        score = 0
        
        if len(self.ts.recent_highs_list) == 0 and len(self.ts.recent_lows_list) == 0:
            self.ts.identifyRecentHighLows (plot=False)
        
        arr = np.array(self.ts.recent_highs_list)
        b = arr[:,0]
        b.sort ()
        
        aux1 = self.ts.time_series.values ()[np.int(b[0])]
        
        for item in b[1:]:
            aux2 = self.ts.time_series.values ()[np.int(item)]
            if aux2 >= aux1:
                score += 1
            else:
                score -= 1
            aux1 = aux2
            
        arr = np.array(self.ts.recent_lows_list)
        b = arr[:,0]
        b.sort ()
        
        aux1 = self.ts.time_series.values ()[np.int(b[0])]
        
        for item in b[1:]:
            aux2 = self.ts.time_series.values ()[np.int(item)]
            if aux2 >= aux1:
                score += 1
            else:
                score -= 1
            aux1 = aux2
            
        return score
        
    #focus on the last 5 recent highs and lows, if each is higher than its predecessors +1, else -1    
    def getHHHLScoreV2 (self, nlast_highs_lows=4):
        
        score_hhhl = 0
        score_lhll = 0 
        
        if len(self.ts.recent_highs_list) == 0 and len(self.ts.recent_lows_list) == 0:
            self.ts.identifyRecentHighLows (plot=False)
        
        arr = np.array(self.ts.recent_highs_list)
        b = arr[:,0]
        b.sort ()
        
        arr = np.array(self.ts.recent_lows_list)
        c = arr[:,0]
        c.sort ()
        
        if len(b) < nlast_highs_lows or len(c) < nlast_highs_lows:
            return 0, 0
        
        aux_high1 = list(self.ts.time_series.values ())[np.int(b[-nlast_highs_lows])]
        aux_low1 = list(self.ts.time_series.values ())[np.int(c[-nlast_highs_lows])]

        aux_high3 = copy.deepcopy(aux_high1)
        aux_low3 = copy.deepcopy(aux_low1)
        
        #aux1 stores the highest high
        for j in range (nlast_highs_lows):
            item_h = b[-nlast_highs_lows+j]
            item_l = c[-nlast_highs_lows+j]
            aux_high2 = list(self.ts.time_series.values ())[np.int(item_h)]
            aux_low2 = list(self.ts.time_series.values ())[np.int(item_l)]

            #first deals with higher highs / higher lows
            if aux_high2 >= aux_high1:
                score_hhhl += 1                
                aux_high1 = aux_high2
            else:
                score = np.maximum(score_hhhl-1,0)
            if aux_low2 >= aux_low1:
                score_hhhl += 1
                aux_low1 = copy.deepcopy(aux_low2)
            else:
                score_hhhl = np.maximum(score_hhhl-1,0)
            
                
            #then with lower highs / lower lows
            if aux_high2 <= aux_high3:
                score_lhll += 1                
                aux_high3 = aux_high2
            else:
                score = np.maximum(score_lhll-1,0)
            if aux_low2 <= aux_low3:
                score_lhll += 1
                aux_low3 = copy.deepcopy(aux_low2)
            else:
                score_lhll = np.maximum(score_lhll-1,0)
            
        return score_hhhl, score_lhll

if False:
    from Trading.Dataset.Dataset import *
    
    ds = Dataset(featpath=r'./datasets/Fx/Featured/NotNormalizedNoVolume/', lookback_window=2)
    ds.last = 2000
    ds.cv_set_size = 1000
    ds.test_set_size = 2000
    ds.loadSeriesByNo(2, bRelabel=False, bNormalize=False, bConvolveCdl=False)

    for relevant_threshold in [20]:
        x = ds.cv_X[:,-1,0]
        x_rsi = ds.cv_X[:,-1,71] 
    
        score_list_lhll = []
        score_list_hhhl = []
        
        rsi_score_list_hhhl = []
        rsi_score_list_lhll = []
        dist_standing_high_list = []
        dist_standing_low_list = []
        dist_relevant_low_list = []
        dist_relevant_high_list = []
        no_standing_highs_list = []
        no_standing_lows_list = []
        no_standing_upward_lines_list = []
        no_standing_downward_lines_list = []
    
        seg_length = 600
        #relevant_threshold = 20
        
        
        for i in range (seg_length, len(x)):
            
            tfe = TrendlineFeaturesExtractor (x=x[i-seg_length:i])
            tfe.ts.identifyRecentHighLows(seg_length,relevant_threshold,False)
            #score_list.append(tfe.getHHHLScore())
            a, b = tfe.getHHHLScoreV2()
            score_list_hhhl.append(a)
            score_list_lhll.append(b)
            
            tfe2 = TrendlineFeaturesExtractor (x=x_rsi[i-seg_length:i])
            tfe2.ts.identifyRecentHighLows(seg_length,relevant_threshold,False)        
            a, b = tfe2.getHHHLScoreV2()
            rsi_score_list_hhhl.append(a)
            rsi_score_list_lhll.append(b)
            
            tfe.ts.checkStandingHighsLows ()
            no_standing_highs_list.append (len(tfe.ts.standing_highs_list))
            no_standing_lows_list.append (len(tfe.ts.standing_lows_list))
            
            tfe.ts.checkStandingLines(True)
            no_standing_upward_lines_list.append (tfe.ts.upward_lines_status_list.count(True))
            tfe.ts.checkStandingLines(False)
            no_standing_downward_lines_list.append (tfe.ts.downward_lines_status_list.count(True))        
            
            a, b, c, d = tfe.getCloserRelevantHighLow (exclude_last_pts=50)
            dist_relevant_high_list.append (c)
            dist_relevant_low_list.append (d)
            
            tfe.ts.checkStandingHighsLows ()
            a, b, c, d = tfe.getCloserStandingHighLow()
            dist_standing_high_list.append (c)
            dist_standing_low_list.append (d)
            
            
        fig, ax1 = plt.subplots (figsize=(14, 10))
        plt.title ("HHHL - "+str(seg_length)+'-'+str(relevant_threshold))
        ax2 = ax1.twinx ()
        ax1.plot (np.linspace(0,len(x)-1, len(x)), x, c='red')
        ax2.plot (np.linspace(seg_length,len(x)-1, len(x)-seg_length), np.array(score_list_hhhl), c='blue')
        ax2.plot (np.linspace(seg_length,len(x)-1, len(x)-seg_length), np.array(rsi_score_list_hhhl), c='brown')
        #ax2.plot (np.linspace(252,len(x)-1, len(x)-252), dist_low_list, c='blue')
        plt.show ()
        
        fig, ax1 = plt.subplots (figsize=(14, 10))
        plt.title ("LHLL - "+str(seg_length)+'-'+str(relevant_threshold))
        ax2 = ax1.twinx ()
        ax1.plot (np.linspace(0,len(x)-1, len(x)), x, c='red')
        ax2.plot (np.linspace(seg_length,len(x)-1, len(x)-seg_length), np.array(score_list_lhll), c='blue')
        ax2.plot (np.linspace(seg_length,len(x)-1, len(x)-seg_length), np.array(rsi_score_list_lhll), c='brown')
        #ax2.plot (np.linspace(252,len(x)-1, len(x)-252), dist_low_list, c='blue')
        plt.show ()
    
