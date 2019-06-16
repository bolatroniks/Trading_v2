# -*- coding: utf-8 -*-

from Framework.Genetic.Functions.threshold_inverters import *
from Framework.Genetic.Functions.feature_functions import *
from Framework.Genetic.Functions.predictors import *

from Framework.Genetic.Gene import *
from Framework.Genetic.GenePCA import GenePCA
from Framework.Dataset.Dataset import Dataset, has_suffix
from Framework.Genetic.utils import prepare_dict_to_save, adapt_dict_loaded

from Framework.Miscellaneous.my_utils import parse_kwargs

from Config.const_and_paths import *
from Framework.Cache.CacheManager import CacheManager
from hashlib import sha1

import numpy as np
import json
import os

class Chromossome ():
    res = {}
    
    def __init__ (self, name = None, ds = None, bDebug = False, bSlim = False,
                  path = CONFIG_TEST_CHROMOSSOME_PATH):
        self.name = name
        self.ds = ds
        self.bDebug = bDebug
        self.path = path
        self.genes = {'full_list':[]}
        for tf in TF_LIST:
            self.genes[tf] = []
            
        self.add_gene (ds = self.ds) #dummy gene
            
    def to_dict (self):
        outdict = {}
        
        for i, g in enumerate(self.genes['full_list']):
            outdict [g.gene_id] = g.to_dict ()
            
        return outdict
            
    def save (self, filename = None, path = None):
        if path is not None:
            self.path = path
        if filename is None:
            filename = self.name + '.crx'
        
        f = open (os.path.join (self.path, filename), 'w')
        f.write (json.dumps(prepare_dict_to_save(self.to_dict ())))
        f.close ()
        
    def load (self, filename, path = None):
        self.clear_genes ()
        
        if path is not None:
            self.path = path        
        
        f = open (os.path.join (self.path, filename), 'r')
        
        chromossome_dict = adapt_dict_loaded(json.loads(f.read ()))
        
        for k, v in chromossome_dict.iteritems ():
            if k.find ('gene') >= 0:
                self.add_gene (None, **v)
                
        for g in self.genes['full_list']:
            g.ds.init_param (instrument = self.ds.ccy_pair,
                             timeframe = g.ds.timeframe,
                             from_time = self.ds.from_time,
                             to_time = self.ds.to_time)
    
    def get_last_fast_timeframe_gene (self):
        for tf in TF_LIST:
            if len(self.genes[tf]) > 0:
                return self.genes[tf][-1]
        
        return None
    
    def get_last_slow_timeframe_gene (self):
        for tf in TF_LIST[-1::-1]:
            if len(self.genes[tf]) > 0:
                return self.genes[tf][-1]
        
        return None
    
    def get_last_timeframe_gene (self, tf):
        if len(self.genes[tf]) > 0:
            return self.genes[tf][-1]
        
        return None
    
    def clear_genes (self):
        for k, v in self.genes.iteritems ():
            self.genes [k] = []
    
    def clear_genes_data (self):
        self.ds.df = None
        self.ds.f_df = None
        self.ds.l_df = None
        self.ds.p_df = None        
        
        for g in self.genes['full_list']:
            g.clear_data ()
            
    def add_gene (self, gene = None,            
                  **kwargs):
    
        if gene is not None:
            self.genes['full_list'].append (gene)
            self.genes[gene.timeframe].append (gene)
            
            if self.ds is None:
                self.ds = gene.ds
        else:
            gene_id = parse_kwargs (['id', 'gene_id'], None, **kwargs)
            timeframe = parse_kwargs (['timeframe', 'tf'], None, **kwargs)
            gene_type = parse_kwargs ('gene_type', 'default', **kwargs)
            ds = parse_kwargs (['ds', 'dataset', 'Dataset'], None, **kwargs)
            func_dict = parse_kwargs ('func_dict', {}, **kwargs)
            pred_type = parse_kwargs ('pred_type', None, **kwargs)
            pred_label= parse_kwargs ('pred_label', 'pred_' + str (len (self.genes['full_list'])), **kwargs) 
            pred_func = parse_kwargs ('pred_func', None, **kwargs) 
            pred_kwargs = parse_kwargs ('pred_kwargs', {}, **kwargs)
            feat_to_keep = parse_kwargs ('feat_to_keep', [], **kwargs)
            status = parse_kwargs ('status', True, **kwargs)
            
            if timeframe is None:
                if ds is not None:
                    timeframe = ds.timeframe
                else:
                    g = self.get_last_fast_timeframe_gene ()
                    if g is not None:
                        timeframe = g.ds.timeframe
                    else:
                        timeframe = 'M15'
                        
            if ds is None:
                if timeframe is not None:
                    g = self.get_last_timeframe_gene (timeframe)
                    if g is not None:
                        ds = g.ds
                    else:
                        if timeframe == self.ds.timeframe:
                            ds = self.ds
                        else:
                            ds = Dataset (ccy_pair = self.ds.ccy_pair,
                                          from_time = self.ds.from_time,
                                          to_time = self.ds.to_time,
                                          timeframe = timeframe)
                else:
                    ds = self.get_last_fast_timeframe_gene (timeframe).ds
            if ds is None:
                raise Exception ('Dataset object cannot be None')
                        
            if gene_type == 'pca':
                print ('method add gene: ' + str (ds))
                self.add_gene(GenePCA (ds = ds, 
                                        gene_id = gene_id,                                        
                                        func_dict = func_dict,
                                        pred_type = pred_type,
                                        status = status,
                                        pred_label = pred_label,
                                        pred_func = pred_func,
                                        pred_kwargs = pred_kwargs
                                    ))
            
            if gene_type == 'slim' or gene_type == 'slim_dataset':
                try:
                    if len(feat_to_keep) == 0:
                        feat_to_keep = parse_kwargs ('feat_to_keep', [], **kwargs['func_dict']['dummy']['kwargs'])
                except:
                    pass
                
                
                if len(feat_to_keep) > 0:
                    func_dict['dummy'] ={'func':slim_dataset,
                                         'kwargs':{'feat_to_keep' : feat_to_keep}}
                                        
                    self.add_gene(Gene (ds = ds,
                                        gene_id = gene_id,
                                        gene_type = gene_type,
                                        func_dict = func_dict,
                                        pred_type = pred_type,
                                        status = status,
                                        pred_label = pred_label,
                                        pred_func = pred_func,
                                        pred_kwargs = pred_kwargs
                                    ))
            
            if gene_type == 'merge':
                daily_delay = parse_kwargs (['daily_delay', 'delay'], 1, **kwargs)
                bConvolveCdl = parse_kwargs (['bConvolveCdl', 'convolve_cdl', 'bConvolveCDL', 'convolveCdl'], True, **kwargs)
                
                slow_tf = parse_kwargs (['slow_tf', 'slow_timeframe', 'higher_tf', 'higher_timeframe'], None, **kwargs)
                
                if slow_tf is None:
                    slow_tf_ds = self.get_last_slow_timeframe_gene ().ds
                    
                else:
                    slow_tf_ds = Dataset (ccy_pair = ds.ccy_pair,
                                           to_time = ds.to_time,
                                           from_time = get_from_time(ds.df.index[0], timeframe),
                                           timeframe = slow_tf
                                           )
                    
                slow_tf = slow_tf_ds.timeframe
                
                for g in self.get_genes_in_proper_order ():
                    for indic in [_ for _ in g.pred_kwargs.keys () if _.find ('indic') >= 0]:
                        if g.pred_kwargs[indic].find ('_' + g.timeframe) < 0:
                            g.pred_kwargs[indic] += '_' + g.timeframe
                            
                    if g.pred_label is not None:
                        if g.pred_label != '':
                            if g.pred_label.find ('_' + g.timeframe) < 0:
                                g.pred_label += '_' + g.timeframe
                            
                self.add_gene (Gene (ds = ds,
                                     gene_id = gene_id,
                                     gene_type = gene_type,
                                     pred_type = pred_type,
                                     status = status,
                                     func_dict={'dummy':
                                                     {'func':merge_timeframes,
                                                      'kwargs':{'ds_slow' : slow_tf_ds,
                                                      'slow_tf': slow_tf,
                                                      'daily_delay': daily_delay,
                                                      'bConvolveCdl' : bConvolveCdl}}}))
                
            if gene_type == 'default':
                if ds is None:
                    g = self.get_last_timeframe_gene (timeframe)
                    
                    if g is not None:
                        ds = g.ds
                    else:                         
                        idx = TF_LIST.index (timeframe)
                        fast_tf_gene = None
                        
                        while (fast_tf_gene is None) and idx > 0:
                            idx -= 1
                            fast_tf_gene = self.get_last_timeframe_gene (TF_LIST[idx])
                        if fast_tf_gene is None:
                            ds=self.ds
                        else:
                            ds = Dataset (ccy_pair = fast_tf_gene.ds.ccy_pair,
                                                               to_time = fast_tf_gene.ds.to_time,
                                                               from_time = get_from_time(fast_tf_gene.ds.df.index[0], timeframe),
                                                               timeframe = timeframe
                                                               )
                if ds is None:
                    raise Exception ('Initialize dataset')
                    
                self.add_gene (Gene (ds = ds, 
                                    gene_id = gene_id,
                                    gene_type = gene_type,
                                    func_dict = func_dict,
                                    pred_type = pred_type,
                                    status = status,
                                    pred_label = pred_label,
                                    pred_func = pred_func, 
                                    pred_kwargs = pred_kwargs,
                                    feat_to_keep = feat_to_keep))
                        
        
    def remove_gene (self, gene_id = None, gene = None):
        if gene_id is not None:
            sel = [_ for _ in self.genes['full_list'] if _.gene_id == gene_id]
            if len (sel) > 0:
                for gene in sel:
                    self.genes['full_list'].remove (gene)
                    self.genes[gene.timeframe].remove (gene)
        else:
            self.genes['full_list'].remove (gene)
            self.genes[gene.timeframe].remove (gene)
        return self
    
    def get_gene (self, gene_id = None):
        if gene_id is None:
            return
        sel = [_ for _ in self.genes['full_list'] if _.gene_id == gene_id]
        if len (sel) > 0:
            return sel [0]
        
        return
        
        
    def get_genes_in_proper_order (self):
        ret = []
        
        ret += [g for g in self.genes['full_list'] if g.gene_type == 'default' and g.timeframe == self.ds.timeframe]
        ret += [g for g in self.genes['full_list'] if g.gene_type == 'default' and g.timeframe != self.ds.timeframe]
        ret += [g for g in self.genes['full_list'] if g.gene_type != 'default' and g.gene_type != 'slim']
        ret += [g for g in self.genes['full_list'] if g.gene_type == 'slim']
        
        return ret
    
    def run_features (self, instrument = None):
        if instrument is not None:
            self.ds.ccy_pair = instrument
            self.ds.loadCandles ()
            
        #this needs to be better designed
        if self.ds.f_df is None:
            self.ds.computeFeatures ()
        if self.ds.l_df is None:
            self.ds.computeLabels (min_stop=0.015)
            
        self.computed_feat_timeframes_list = [self.ds.timeframe]
                
        for g in self.get_genes_in_proper_order ():
            d = {'gene': prepare_dict_to_save(g.to_dict ()),
                     'instrument': self.ds.ccy_pair,
                     'timeframe': self.ds.timeframe,
                     'from_time': self.ds.from_time,
                     'to_time': self.ds.to_time}
            del (d['gene']['pred_type']) #this info is not need when caching and retrieving and can cause problems
            del (d['gene']['status']) #this info is not need when caching and retrieving and can cause problems
            print ('cache lookup: ' + str (d))
            if CacheManager.get_cached_object (sha1(str (d)).hexdigest ()) is not None:
                print ('gene prediction found in cache')
                g.bInCache = True
                continue
            
            if instrument is not None:
                g.ds.init_param (instrument = instrument,
                                 timeframe = g.ds.timeframe,
                                 from_time = g.ds.from_time,
                                 to_time = g.ds.to_time)
                g.load_data ()
            
            g.compute_added_features ()
            
            self.computed_feat_timeframes_list.append (g.ds.timeframe)
                            
            if self.bDebug:
                try:
                    print (g.to_dict ())
                    print ('df: ' + str (self.ds.df.shape) )
                    print ('f_df: ' + str (self.ds.f_df.shape) )
                    print ('l_df: ' + str (self.ds.l_df.shape) )
                    print ('p_df: ' + str (self.ds.p_df.shape) )
                except Exception as e:
                    print (e.message)
        
    def run_features_old (self, instrument = None):
        if instrument is not None:
            self.ds.ccy_pair = instrument
            self.ds.loadCandles ()
            
        #this needs to be better designed
        if self.ds.f_df is None:
            self.ds.computeFeatures ()
        if self.ds.l_df is None:
            self.ds.computeLabels (min_stop=0.015)
                
        for g in self.get_genes_in_proper_order ():
            if instrument is not None:
                g.ds.init_param (instrument = instrument,
                                 timeframe = g.ds.timeframe,
                                 from_time = g.ds.from_time,
                                 to_time = g.ds.to_time)
                g.load_data ()
            
            g.compute_added_features ()
                            
            if self.bDebug:
                try:
                    print (g.to_dict ())
                    print ('df: ' + str (self.ds.df.shape) )
                    print ('f_df: ' + str (self.ds.f_df.shape) )
                    print ('l_df: ' + str (self.ds.l_df.shape) )
                    print ('p_df: ' + str (self.ds.p_df.shape) )
                except Exception as e:
                    print (e.message)
                    
    def run_predictions (self):
        #resets predictions to avoid trouble down the line
        self.ds.set_predictions ()
        
        #just in case, better add suffixes whether or not chromossome uses more than 1 timeframe
        self.add_suffixes ()
        
        for i, g in enumerate(self.genes['full_list']):
            if g.pred_func is not None:
                g.compute_predictions (ds = self.ds)
        self.aggregate_predictions ()
                
    def aggregate_predictions (self):
        direc_pred = np.zeros (len (self.ds.p_df))
        binary_pred = np.zeros (len (self.ds.p_df))
        stop_buy_pred = np.zeros (len (self.ds.p_df))
        stop_sell_pred = np.zeros (len (self.ds.p_df))
        
        binary_threshold = 0
        direc_threshold = 0
        
        for g in self.genes['full_list']:
            if g.pred_func is not None and g.status:
                if 'pred_type' in g.pred_kwargs:
                    g.pred_type = g.pred_kwargs['pred_type']
                    
                if g.pred_type is not None:
                    if g.pred_type == 'binary':
                        binary_pred += self.ds.p_df[g.pred_label]
                        binary_threshold += 1
                    if g.pred_type == 'directional':
                        direc_pred += self.ds.p_df[g.pred_label]
                        direc_threshold += 1
                    if g.pred_type == 'preventer': #lack of better name
                        #this prevents buying or selling
                        stop_buy_pred += self.ds.p_df[g.pred_label + '_not_buy']
                        stop_sell_pred += self.ds.p_df[g.pred_label + '_not_sell']
                        
                else:
                    if 'inv_threshold_fn' in g.pred_kwargs:
                        k = 'inv_threshold_fn'
                    elif 'inv_threshold_fn1' in g.pred_kwargs:
                        k = 'inv_threshold_fn1'
                    elif 'inv_threshold_fn2' in g.pred_kwargs:
                        k = 'inv_threshold_fn2'
                    else:
                        k = None
                    if k is not None:                    
                        if g.pred_kwargs[k] is not None:
                            pred_type = 'directional'
                            direc_pred += self.ds.p_df[g.pred_label]
                            direc_threshold += 1
                        else:
                            pred_type = 'binary'
                            binary_pred += self.ds.p_df[g.pred_label]
                            binary_threshold += 1
                    else:
                        pred_type = 'binary'
                        binary_pred += self.ds.p_df[g.pred_label]
                        binary_threshold += 1
                        
                    g.pred_type = pred_type
                    print g.pred_label + ' pred type: ' + pred_type
                
        self.ds.p_df.Predictions = (binary_pred >= binary_threshold) * \
                                     (direc_pred * direc_pred >= direc_threshold * direc_threshold) * \
                                     np.sign (direc_pred)
        #preventers kick in
        self.ds.p_df.Predictions = np.maximum(self.ds.p_df.Predictions, NEUTRAL_SIGNAL) * (stop_buy_pred == 0) + \
                                    np.minimum(self.ds.p_df.Predictions, NEUTRAL_SIGNAL) * (stop_sell_pred == 0)
            
    def run (self, instrument = None, 
                     from_time = None, 
                     to_time = None):
        
        #tests if the chromossome is empty
        if len (self.get_genes_in_proper_order ()) == 0:
            return
        
        if instrument is not None:
            if instrument != self.get_last_fast_timeframe_gene ().ds.ccy_pair:
                self.clear_genes_data ()
                
        #removes dummy gene to avoid trouble
        genes_to_delete = [_ for _ in self.get_genes_in_proper_order () if _.pred_func is None and _.func_dict == {}]
        
        for gene_to_delete in genes_to_delete:
            self.remove_gene (gene_id = gene_to_delete.gene_id)
            del gene_to_delete #release memory
                
        #tests if the chromossome is empty again after removing dummies
        if len (self.get_genes_in_proper_order ()) == 0:
            return
                
        #makes self.ds equal to the fastest timeframe
        #this will cause trouble
        self.ds = self.get_last_fast_timeframe_gene ().ds
                
        if type(instrument) == list:
            for _ in instrument:
                self.run (instrument = _,
                          from_time = from_time,
                          to_time = to_time)
    
                stats = self.get_stats ()         
                for k in stats.keys ():
                    try:
                        stats[k]['gene_content'] = prepare_dict_to_save (self.get_gene (k).to_dict ())
                    except:
                        pass
                
                Chromossome.res [self.ds.ccy_pair + '_' + self.ds.timeframe + '_' +  
                     self.ds.from_time + '_' + self.ds.to_time] = {'len': len(self.ds.f_df),
                                                                     'stats': stats}
        else:
            #if from_time is not None or to_time is not None:
            #    if instrument != self.get_last_fast_timeframe_gene ().timeframe:
            #        self.clear_genes_data ()
                    
            self.run_features (instrument = instrument)
            
            #checks if there is more than one timeframe in the chromossome
            if len (list(set (self.computed_feat_timeframes_list))) > 1:
                #checks if there is a merge gene
                #if so, merges timeframes
                self.merge_timeframes ()
                            #deprecated
                            #if 'merge' not in [g.gene_type for g in self.genes['full_list']]:
                                #self.add_gene (gene_type = 'merge')
                        
            self.run_predictions ()
            
    def add_suffixes (self):
        #adds timeframe as suffix to feature names
        self.ds.f_df.columns = [col + ('_' + self.ds.timeframe if not has_suffix (col) else '') for col in self.ds.f_df.columns]
                    
        for g in self.get_genes_in_proper_order ():
            for indic in [_ for _ in g.pred_kwargs.keys () if _.find ('indic') >= 0]:
                if g.pred_kwargs[indic].find ('_' + g.timeframe) < 0:
                    g.pred_kwargs[indic] += '_' + g.timeframe
                    
            if g.pred_label is not None:
                if g.pred_label != '':
                    if g.pred_label.find ('_' + g.timeframe) < 0:
                        g.pred_label += '_' + g.timeframe
        
        
    def merge_timeframes (self):
        
        try:
            self.dsh
        except:
            self.dsh = None
        if self.dsh is None:
            self.dsh = DatasetHolder(instrument = self.ds.ccy_pair,
                            from_time=self.ds.from_time, to_time=self.ds.to_time)
        tf_list = list(set ([g.timeframe for g in self.genes['full_list']]))
        for tf in tf_list:
            self.dsh.ds_dict [self.ds.ccy_pair + '_' + tf] = self.get_last_timeframe_gene (tf).ds
        
        for tf in tf_list:
            if tf != self.get_last_fast_timeframe_gene ().timeframe:
                self.dsh.appendTimeframesIntoOneDataset (
                        instrument = self.ds.ccy_pair,
                        higher_timeframe = tf,
                         lower_timeframe = self.get_last_fast_timeframe_gene ().timeframe,
                         daily_delay = 1, 
                         bConvolveCdl = True)
                
        #adds suffixes to features and indicators used in the predictions
        self.add_suffixes ()
        
        return self
        
    def get_stats (self):
        self.run ()
        
        #edge case whereby chromossome is empty
        if len (self.get_genes_in_proper_order ()) == 0:
            return {'Overall': {'hit_ratio': 0.0,
                      'longs': 0.0,
                      'ret_10': 0.0,
                      'ret_25': 0.0,
                      'ret_50': 0.0,
                      'shorts': 0.0}}
        
        if self.ds.l_df is None or len (self.ds.p_df) != len (self.ds.l_df):
            self.ds.computeLabels (min_stop = 0.015) #TODO: need to improve this
        
        if len (self.ds.p_df) >= len (self.ds.l_df):
            preds = self.ds.p_df.Predictions [-len (self.ds.l_df):]
            labels = self.ds.l_df.Labels
            ret10 = self.ds.l_df.ret_10_periods
            ret25 = self.ds.l_df.ret_25_periods
            ret50 = self.ds.l_df.ret_50_periods
        else:
            preds = self.ds.p_df.Predictions
            labels = self.ds.l_df.Labels  [-len (self.ds.p_df):]
            ret10 = self.ds.l_df.ret_10_periods [-len (self.ds.p_df):]
            ret25 = self.ds.l_df.ret_25_periods [-len (self.ds.p_df):]
            ret50 = self.ds.l_df.ret_50_periods [-len (self.ds.p_df):]
            
        stats_dict = {'Overall': {
                                    'hit_ratio': np.float(np.count_nonzero( 
                                    (preds == labels) & (preds != NEUTRAL_SIGNAL)
                                    )) / (np.float(np.count_nonzero(preds != NEUTRAL_SIGNAL)) + 0.000001),
                                    'longs': np.float(np.count_nonzero (preds == 1)) / np.float (len(preds)),
                                    'shorts': np.float(np.count_nonzero (preds == -1)) / np.float (len(preds)),
                                    'ret_10': np.sum (preds * ret10) / (np.float(np.count_nonzero(preds != NEUTRAL_SIGNAL)) + 0.000001),
                                    'ret_25': np.sum (preds * ret25) / (np.float(np.count_nonzero(preds != NEUTRAL_SIGNAL)) + 0.000001),
                                    'ret_50': np.sum (preds * ret50) / (np.float(np.count_nonzero(preds != NEUTRAL_SIGNAL)) + 0.000001)
                                }}
        
        for i, g in enumerate(self.genes['full_list']):
            stats_dict [g.gene_id] = {'status': g.status}
            if 'pred_type' in g.pred_kwargs:
                pred_type = g.pred_kwargs ['pred_type']
                
                stats_dict [g.gene_id] ['pred_type'] = pred_type
                if pred_type == 'preventer':
                    stats_dict [g.gene_id] ['not buy'] = np.count_nonzero (self.ds.p_df[g.pred_label + '_not_buy']) / np.float (len(self.ds.p_df))
                    stats_dict [g.gene_id] ['not sell'] = np.count_nonzero (self.ds.p_df[g.pred_label + '_not_sell']) / np.float (len(self.ds.p_df))
            else:
                if g.pred_label in self.ds.p_df.columns:                
                    if g.pred_label.find ('pred:') >= 0:
                        
                        stats_dict [g.gene_id] = {                                                
                                                'status': g.status,
                                                'label':g.pred_label,
                                                'longs': np.float(np.count_nonzero (self.ds.p_df[g.pred_label] == 1)) / np.float (len(self.ds.p_df)),
                                                'shorts': np.float(np.count_nonzero (self.ds.p_df[g.pred_label] == -1)) / np.float (len(self.ds.p_df))
                                               }
                        
                        if 'inv_threshold_fn' in g.pred_kwargs:
                            k = 'inv_threshold_fn'
                        elif 'inv_threshold_fn1' in g.pred_kwargs:
                            k = 'inv_threshold_fn1'
                        elif 'inv_threshold_fn2' in g.pred_kwargs:
                            k = 'inv_threshold_fn2'
                        else:
                            k = None
                        if k is not None:
                            if g.pred_kwargs [k] is not None:
                                stats_dict [g.gene_id] ['hit_ratio'] = np.float(np.count_nonzero( 
                                        (self.ds.p_df[g.pred_label] == self.ds.l_df.Labels) & (self.ds.p_df[g.pred_label] != NEUTRAL_SIGNAL)
                                        )) / (np.float(np.count_nonzero(self.ds.p_df[g.pred_label] != NEUTRAL_SIGNAL)) + 0.0000001)
                                stats_dict [g.gene_id] ['ret_10'] = np.sum (self.ds.p_df[g.pred_label] * self.ds.l_df.ret_10_periods) / (np.float(np.count_nonzero(self.ds.p_df[g.pred_label] != NEUTRAL_SIGNAL)) + 0.000001)
                                stats_dict [g.gene_id] ['ret_25'] = np.sum (self.ds.p_df[g.pred_label] * self.ds.l_df.ret_25_periods) / (np.float(np.count_nonzero(self.ds.p_df[g.pred_label] != NEUTRAL_SIGNAL)) + 0.000001)
                                stats_dict [g.gene_id] ['ret_50'] = np.sum (self.ds.p_df[g.pred_label] * self.ds.l_df.ret_50_periods) / (np.float(np.count_nonzero(self.ds.p_df[g.pred_label] != NEUTRAL_SIGNAL)) + 0.000001)
                            else:
                                 stats_dict [g.gene_id] ['active'] = np.float(np.count_nonzero (self.ds.p_df[g.pred_label] != NEUTRAL_SIGNAL)) / np.float (len(self.ds.p_df))
        return stats_dict
    
    def save_stats (self, filename = 'stats_test.txt'):
        stats_dict = self.get_stats ()
        
        f = open (os.path.join (self.path, filename), 'a')
        for k, v in stats_dict.iteritems ():
            outstr = self.ds.ccy_pair + ', ' + str(self.ds.from_time) + ', ' + str(self.ds.to_time) + ', ' + self.ds.timeframe + ',' + str (len(self.ds.p_df)) + ',' + k + ', '
            if self.get_gene(k) is not None:
                outstr += str (prepare_dict_to_save(self.get_gene(k).to_dict ())).replace (',', '_') + ','
            else:
                outstr += 'NA,'
            
            for _ in ['longs', 'shorts', 'hit_ratio', 'ret_10', 'ret_25', 'ret_50']:
                if _ in v.keys ():
                    outstr += str(v [_]) + ','
                else:
                    outstr += str ('NA') + ','
            
            f.write (outstr + '\n')
        f.close
        
    def get_permutations (self):
        permutations = []
        
        #base case
        if len ([_ for _ in self.get_genes_in_proper_order () if _.pred_func is not None or _.func_dict != {}]) == 0: 
            return [self]
        
        #recursion
        for g in [_ for _ in self.get_genes_in_proper_order () if _.pred_func is not None or _.func_dict != {}]:
            crx1 = self.clone ()                        
            crx1.remove_gene (gene_id = g.gene_id)
            l1 = crx1.get_permutations ()
            
            l2 = [_.clone () for _ in l1]
            [_.add_gene (g) for _ in l2]
        
        return permutations + l1 + l2
            
            
    def clone (self):
        crx = Chromossome (ds = self.ds)
        
        #removes the dummy gene, so that it doesn't get copied over and over
        genes_to_delete = [_ for _ in crx.get_genes_in_proper_order () if _.pred_func is None and _.func_dict == {}]
        
        for gene_to_delete in genes_to_delete:
            crx.remove_gene (gene_id = gene_to_delete.gene_id)
            del gene_to_delete #release memory
        for g in self.get_genes_in_proper_order ():
            crx.add_gene (g)
            
        return crx
            
            
                        