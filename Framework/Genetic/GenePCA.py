# -*- coding: utf-8 -*-

from Trading.Genetic.Gene import Gene
from Trading.FeatureExtractors.Model.CrossSection.PCA import PCA
from Framework.Cache import CacheManager
from Config.const_and_paths import full_instrument_list

class GenePCA (Gene):
    def __init__ (self, ds, gene_id = None,
                  func_dict = {},
                  pred_type = None,
                  status = True,
                  pred_label = '',
                  pred_func = None, 
                  pred_kwargs = {},
                  pca = None,
                  pca_feat_path = None):
        
        print ('Inside the constructor of  GenePCA: ' + str (ds))
        if pca_feat_path is not None:
            ds.pca_feat_path = pca_feat_path
        
        Gene.__init__ (self, ds = ds, gene_id = gene_id,
                       gene_type = 'pca',
                       func_dict = func_dict, 
                       pred_type = pred_type,
                       status = status,
                       pred_label = pred_label,
                       pred_func = pred_func,
                       pred_kwargs = pred_kwargs
                       )
        self.pca = pca
        self.load_data ()
        
    def load_data (self):
        Gene.load_data (self)
        
        if self.pca is None:            
            self.pca = CacheManager.get_cached_object('pca')
            
            if self.pca is None:
                self.pca = PCA(from_time = self.ds.from_time,
                          to_time=self.ds.to_time, 
                          bLoadCandlesOnline=self.ds.bLoadCandlesOnline, 
                          instrument_list=full_instrument_list)
        
        return self
    
    def compute_added_features (self, func_dict = {}):
        if self.ds is None:
            raise Exception ('Dataset object cannot be None')
            
        if self.ds.f_df is None:
            self.ds.computeFeatures(bComputeHighLowFeatures=False)
        
        if self.ds.bLoadCandlesOnline:
            if self.pca.pca_df.index [-1] < self.ds.f_df.index [-1]:                
                self.pca.to_time = self.ds.to_time
                
                self.pca.load_input_data ()
                self.pca.extract_principal_components ()
                CacheManager.cache_object (name='pca', obj = self.pca)                
                
            self.pca.compute_pca_features(self.ds)
            self.pca.save_pca_features (self.ds)
            
            for col in self.pca.pca_feat_df.columns:
                self.ds.f_df[col] = self.pca.pca_feat_df[col][self.ds.f_df.index]
        else:
            self.ds.loadPCAFeatures ()
            
        Gene.compute_added_features (self, func_dict)
        
        return self
        

if False:
    g = GenePCA(ds = Dataset(ccy_pair='USD_ZAR', 
                          from_time='2015-10-01 00:00:00', 
                          to_time=2015, 
                          timeframe='M15'))