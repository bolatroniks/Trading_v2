# -*- coding: utf-8 -*-

from copy import deepcopy

from Framework.Genetic.utils import prepare_dict_to_save

from Framework.Genetic.Functions.feature_functions import *

from Miscellaneous.Cache.CacheManager import CacheManager
from hashlib import sha1

import os
import pandas as pd
import json

#ToDo: create sample genes and add them to a dictionary (add set methods that return the gene itself for easy customization)

#TODO: need a more elegant solution for this
B_COMPUTE_HIGH_LOW_FEATURES = False

class Gene ():
    count = 0

    def __init__ (self, ds, gene_id = None, 
                  gene_type = 'default', func_dict = {},
                  pred_type = None,
                  status = True,
                  pred_label = '',
                  pred_func = None, 
                  pred_kwargs = {},
                  feat_to_keep = ['RSI', 'Close']      #keeps a list of all necessary features to compute the added features of this gene and predictions
                        #False means that it'll get rid of all undesired features after computing its added features
                                                #better have only one gene doing that
                  ):
        if gene_id is not None:
            self.gene_id = gene_id
        else:
            self.gene_id = 'gene_' + str (Gene.count)
        
        self.ds = ds
        self.gene_type = gene_type
        
        self.timeframe = self.ds.timeframe
        self.func_dict = func_dict #dictionary of functions to compute new features
        self.pred_label = (pred_label if pred_label.find ('pred:') >= 0 else 'pred:' + pred_label)
        self.pred_func = pred_func
        self.pred_kwargs = pred_kwargs
        self.feat_to_keep = feat_to_keep
        self.feat_computed = False
        self.pred_computed = False
        
        if pred_type is not None:
            self.pred_type = pred_type
        else:
            if 'inv_threshold_fn' in pred_kwargs.keys ():
                if pred_kwargs ['inv_threshold_fn'] is not None:
                    self.pred_type = 'symmetric'
                else:
                    self.pred_type = 'binary'
            elif self.pred_func is not None:
                self.pred_type = 'binary'
            else:
                self.pred_type = 'dummy'

        #TODO: implement the idea below clearly and correctly        
        # there are the following types of genes:
        #   1/ symmetric (None): strategy buys and sells an asset using the same criteria
        #   2/ bynary: when active, allows signals to be generated, when inactive, suppress both buy and sell signals
        #   3/ preventer: when active, prevents either buying, selling or both
        #   4/ asymmetric: for long only or short only strategies
        #           eg.: long SPX at the turn of the month;
        #               long stocks at the turn of the quarter when bonds outperformed massively
        
        self.status = status    #means gene is active if True
                              #just to perform some quick tests suppressing some genes
        
        #Gene.load_data (self)    #better use lazy initialization
        self.bInCache = False
        
        Gene.count += 1
        
    def to_dict (self):        
        out_dict = {}
        out_dict ['gene_type'] = self.gene_type
        out_dict ['timeframe'] = self.timeframe
        
        func_dict = deepcopy (self.func_dict)

        out_dict ['func_dict'] = func_dict
        
        out_dict ['status'] = self.status
        
        out_dict ['pred_type'] = self.pred_type
        
        out_dict ['pred_label'] = self.pred_label

        out_dict ['pred_func'] = self.pred_func
        pred_kwargs = deepcopy (self.pred_kwargs)
        
        try:
            out_dict ['pred_kwargs'] = pred_kwargs
        except Exception as e:
            print e.message
            out_dict ['pred_kwargs'] = {}
                
        return out_dict
    
    def save (self, filename, path = None):
        if path is None:
            path = CONFIG_TEST_GENES
        
        if filename is None:
            filename = self.gene_id + '.crx'
        
        f = open (os.path.join (self.path, filename), 'w')
        f.write (json.dumps(prepare_dict_to_save (self.to_dict ())))
        f.close ()
    
    def load (self, filename, path = None):
        if path is None:
            path = CONFIG_TEST_GENES
            
        f = open (os.path.join (path, filename), 'r')
        
    
    def load_data (self):
        if self.ds is None:
            raise Exception ('Dataset object cannot be None')
        if self.ds.df is None:
            self.ds.loadCandles ()
            
        return self
    
    def clear_data (self):
        self.ds.df = None
        self.ds.f_df = None
        self.ds.l_df = None
        self.ds.p_df = None
        self.feat_computed = False
        self.pred_computed = False
    
    def compute_added_features (self, func_dict = {}):
        YEARS_ADDED_TO_DATASET_TO_COMPUTE_FEATURES_WITH_LARGE_LOOKBACK_WINDOW = 3
        
        if not self.feat_computed:
            for k, v in func_dict.iteritems ():
                if k not in self.func_dict.keys ():
                    self.func_dict[k] = v
            
            if self.ds is None:
                raise Exception ('Dataset object cannot be None')
            
            if self.ds.f_df is None:
                if self.ds.timeframe == 'D' or self.ds.timeframe == 'W':
                    if not self.ds.bLoadCandlesOnline:
                        try:
                            raise Exception
                            #self.ds.loadFeatures ()
                        except:
                            #changes the start data of the dataset, in order to compute features with a large lookback window correctly
                            old_start = pd.Timestamp(self.ds.from_time)
                            new_start = pd.Timestamp ("%4d-%02d-%02d %02d:%02d:%02d" % (old_start.year - YEARS_ADDED_TO_DATASET_TO_COMPUTE_FEATURES_WITH_LARGE_LOOKBACK_WINDOW, 
                                                      old_start.month, 
                                                      old_start.day, 
                                                      old_start.hour, 
                                                      old_start.minute, 
                                                      old_start.second))
                            
                            self.ds.set_from_to_times (from_time = new_start)
                            self.ds.loadCandles ()
                            self.ds.computeFeatures(bSaveFeatures=True, 
                                                    bComputeHighLowFeatures = B_COMPUTE_HIGH_LOW_FEATURES)
                    else:
                        #changes the start data of the dataset, in order to compute features with a large lookback window correctly
                        old_start = pd.Timestamp(self.ds.from_time)
                        new_start = pd.Timestamp (old_start.year - YEARS_ADDED_TO_DATASET_TO_COMPUTE_FEATURES_WITH_LARGE_LOOKBACK_WINDOW, 
                                old_start.month, old_start.day, old_start.hour, old_start.minute, old_start.second)
                        
                        self.ds.set_from_to_times (from_time = new_start)
                        self.ds.loadCandles ()
                        self.ds.computeFeatures(bSaveFeatures=True)
                        
                else:
                    if self.ds.df is None:                        
                        self.ds.loadCandles ()
                    self.ds.computeFeatures()
                
            for feat, func_args in self.func_dict.iteritems ():
                self.ds.f_df [feat] = func_args['func'] (ds=self.ds, 
                                                         **func_args['kwargs'])
               
            self.feat_computed = True
        
        return self
    
    def compute_predictions (self, ds = None, func = None, kwargs = {}):
        if ds is None:            
            ds = self.ds
        else:
            self.ds = ds
        #will be using for caching and retrieving predictions
        d = {'gene': prepare_dict_to_save(self.to_dict ()),
                     'instrument': ds.ccy_pair,
                     'timeframe': ds.timeframe,
                     'from_time': ds.from_time,
                     'to_time': ds.to_time}
        del (d['gene']['pred_type']) #this info is not need when caching and retrieving and can cause problems
        del (d['gene']['status']) #this info is not need when caching and retrieving and can cause problems
        
        cache_attempt = CacheManager.get_cached_object (sha1(str (d)).hexdigest ())
        if cache_attempt is not None:
            print ('gene prediction found in cache')
            self.bInCache = True
            
            if ds.p_df is None:
                ds.set_predictions ()
            
            if self.pred_type == 'preventer':
                ds.p_df [self.pred_label + '_not_buy'] = deepcopy(cache_attempt [0])
                ds.p_df [self.pred_label + '_not_sell'] = deepcopy(cache_attempt [1])
                
            elif self.pred_label is not None:
                print ('\n')
                print (self.pred_label in ds.p_df.columns)
                print (type(cache_attempt))
                print ('\n')
                ds.p_df [self.pred_label] = deepcopy(cache_attempt)
            
            self.pred_computed = True
            return self
        
        #if not self.pred_computed:             
        else:
            if func is not None:
                self.pred_func = func
                
            if len (kwargs.keys()) > 0:
                self.pred_kwargs = kwargs
            
            if ds is None:
                ds = self.ds
                
            if ds is None:
                raise Exception ('Dataset object cannot be None')
            if ds.p_df is None:
                ds.computePredictions ()
                
            if self.pred_func is not None:
                if 'pred_type' in self.pred_kwargs:
                    if self.pred_kwargs['pred_type'] == 'preventer':
                        self.pred_type = 'preventer'
                if self.pred_type == 'preventer':
                    prevent_buy, prevent_sell = self.pred_func (ds, **self.pred_kwargs)
                    
                    ds.p_df [self.pred_label + '_not_buy'] = prevent_buy
                    ds.p_df [self.pred_label + '_not_sell'] = prevent_sell
                    
                    CacheManager.cache_object(
                                name = sha1(str (d)).hexdigest (),
                                obj = deepcopy ( [ds.p_df[self.pred_label + '_not_buy'], 
                                               ds.p_df[self.pred_label  + '_not_sell']]))
                    
                    
                    self.pred_computed = True
                    return self
                        
                #ds.set_predictions (ds.p_df.Predictions + self.pred_func (ds, **self.pred_kwargs))
                ds.p_df[self.pred_label] = self.pred_func (ds, **self.pred_kwargs)
                                
                #caches the predictions                
                CacheManager.cache_object(
                                name = sha1(str (d)).hexdigest (),
                                obj = deepcopy (ds.p_df[self.pred_label]))
                print ('Cached object ' + str (d))
            self.pred_computed = True
            
        return self
    
    def merge_timeframes (self, ds_lower_tf = None, slow_tf = 'D',
                          daily_delay = 1, bConvolveCdl = True):
        if ds_lower_tf is None:
            ds_lower_tf = Dataset(ccy_pair = self.ds.ccy_pair, timeframe = slow_tf, 
                                from_time = get_from_time(self.ds.df.index[-1], slow_tf), 
                                to_time=self.ds.df.index[-1],
                                bLoadCandlesOnline = self.ds.bLoadCandlesOnline)
        if ds_lower_tf.df is None:
            ds_lower_tf.loadCandles ()
        if ds_lower_tf.f_df is None:
            ds_lower_tf.computeFeatures (bComputeHighLowFeatures = False) #high low features are not calculated by default
        
        try:
            self.dsh
        except:
            self.dsh = None
        if self.dsh is None:
            self.dsh = DatasetHolder(instrument = self.ds.ccy_pair,
                            from_time=self.ds.from_time, to_time=self.ds.to_time)
            self.dsh.ds_dict [self.ds.ccy_pair + '_' + self.ds.timeframe] = self.ds
        
        self.dsh.ds_dict [self.ds.ccy_pair + '_' + ds_lower_tf.timeframe] = ds_lower_tf
        
        self.dsh.appendTimeframesIntoOneDataset (higher_timeframe = ds_lower_tf.timeframe,
                                                 lower_timeframe = self.ds.timeframe,
                                                 daily_delay = daily_delay, 
                                                 bConvolveCdl = bConvolveCdl)
        
        return self
    
class GeneSlimDataset (Gene): #only purpose is to get rid of unwanted features
    def __init__ (self, ds, func_dict = {}, pred_func = None, 
                  feat_to_keep = ['RSI', 'Close']):
        Gene.__init__ (self, ds = ds, func_dict = func_dict,
                       pred_func = pred_func)
        self.feat_to_keep = feat_to_keep
        
    def compute_added_features (self):
        
        for col in self.ds.f_df.columns:
            if (col not in self.func_dict.keys()) and (col not in self.feat_to_keep):
                del self.ds.f_df[col]
    
        return self

if False:
    g = Gene(ds = Dataset(ccy_pair='USD_ZAR', 
                          from_time=2013, 
                          to_time=2015, 
                          timeframe='M15'),
            pred_func = fn_pred3,
            pred_kwargs = {'indic': 'RSI',
                                     'threshold_min': 30,
                                     'threshold_max': 40}
                        ).load_data ()
            
    g.compute_added_features (func_dict={'new_hilo': {'func':fn_new_hilo, 'kwargs':{'window':126}}})
    g.compute_predictions ()
    plt.plot(g.ds.p_df.Predictions)
    g.merge_timeframes ()
    g.ds.f_df['CDL3INSID_D'].plot ()