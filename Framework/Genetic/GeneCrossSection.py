# -*- coding: utf-8 -*-

from Framework.Genetic.Gene import Gene
from Framework.Features.CrossSection.PCA.PCA import PCA
from Miscellaneous.Cache.CacheManager import CacheManager
from Config.const_and_paths import full_instrument_list
from Framework.Dataset.DatasetHolder import DatasetHolder
from Framework.Dataset.Dataset import Dataset

class GeneCrossSection (Gene):
    '''
    This class should be used to compute features depending on cross section slices
    as opposed to timeseries.
    
    Eg.:    asset correlation with SPX > 0.6;
            asset lagging another.   
    '''
    
    def __init__ (self, ds, gene_id = None,
                  func_dict = {},
                  cross_section_func_dict = {},
                  pred_type = None,
                  status = True,
                  pred_label = '',
                  pred_func = None, 
                  pred_kwargs = {},
                  ds_holder = None,
                  feat_path = None,
                  bSaveFeats = False #to be used for feats requiring heavy computations, eg: PCA
                  ):
        
        print ('Inside the constructor of  GeneCrossSection: ' + str (ds))
        #if pca_feat_path is not None:
        #    ds.pca_feat_path = pca_feat_path
        
        Gene.__init__ (self, ds = ds, gene_id = gene_id,
                       gene_type = 'cross_section',
                       func_dict = func_dict, 
                       pred_type = pred_type,
                       status = status,
                       pred_label = pred_label,
                       pred_func = pred_func,
                       pred_kwargs = pred_kwargs
                       )
        self.ds_holder = ds_holder
        self.cross_section_func_dict = cross_section_func_dict
        self.load_data ()
        
    def load_data (self):
        Gene.load_data (self)
        
        if self.ds_holder is None:            
            self.ds_holder = CacheManager.get_cached_object('cross_section_' + self.gene_id)
            
        return self
    
    def compute_added_features (self, func_dict = {}):
        if self.ds is None:
            raise Exception ('Dataset object cannot be None')
            
        if self.ds.f_df is None:
            self.ds.computeFeatures(bComputeHighLowFeatures=False)
        
        if self.ds_holder is None:
            raise Exception ('DatasetHolder object cannot be None')
            
        #computes features not depending on cross-section slice            
        Gene.compute_added_features (self, self.func_dict)
        
        #computes cross-section features
        for feat, func_args in self.cross_section_func_dict.iteritems ():
            self.ds.f_df [feat] = func_args['func'] (ds=self.ds, 
                                                     ds_holder = self.ds_holder,
                                                     **func_args['kwargs'])
               
            
        
        return self
        

if True:
    #In this test, we will compute the correlation between USDZAR and SPX, UST 10Y and EURUSD
    ds = Dataset (ccy_pair = 'USD_ZAR', 
                  timeframe = 'H1', 
                  from_time = 2010, to_time = 2011, 
                  bLoadCandlesOnline = False)
    
    dsh = DatasetHolder(from_time = ds.from_time, to_time = ds.to_time)
    dsh.loadMultiFrame (timeframe_list=[ds.timeframe], 
                        ccy_pair_list=['EUR_USD', 'SPX500_USD'],
                        bComputeFeatures = [True, True])
    
    #TODO: compute correlations
    #check how correlations are computed in PCA features
    
    
    