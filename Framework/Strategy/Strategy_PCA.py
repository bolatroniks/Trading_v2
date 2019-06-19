# -*- coding: utf-8 -*-
from Framework.Features.CrossSection.PCA.PCA import PCA
from Framework.Features.TimeSeries import halflife
from Framework.Strategy.Strategy import Strategy
from Framework.Dataset.DatasetHolder import DatasetHolder
from Framework.Dataset.Dataset import Dataset
from Framework.Dataset.dataset_func import get_from_time
from Config.const_and_paths import fx_list, full_instrument_list, NEUTRAL_SIGNAL
from Logging import LogManager

import numpy as np
from copy import deepcopy


#filename = os.path.join(CONFIG_PROD_RULE_PATH, 'MTF_PCA', 'default.stp')
#f = open (filename, 'r')
#kwargs = eval(f.read ())

# =============================================================================
# Rule(name='prod_mtf_pca', 
#                               func = mtf_pca, 
#                               args=kwargs, 
#                               ruleType='MultiTimeframe', 
#                               target_multiple=1.5,
#                               bUseHighLowFeatures= True,
#                               timeframe = 'M15',
#                               other_timeframes = ['D'])
# =============================================================================

class Strategy_PCA (Strategy):
    def __init__ (self, name='My_Strat', 
                  rule = None, 
                  instruments=fx_list,
                  reporting_ccy = 'USD',
                  value_per_bet = 300, 
                  max_open_positions=5, 
                  serial_gap=20):
                      
        Strategy.__init__ (self, name=name, rule = rule, instruments = instruments,
                  reporting_ccy = reporting_ccy, value_per_bet = value_per_bet,
                  max_open_positions = max_open_positions,
                  serial_gap = serial_gap)
        
    def compute_additional_features (self):
        aux_ds = [_ for _ in self.other_ds if _.timeframe == 'D'][0]
        b = np.array([halflife (aux_ds.f_df.Close[i-252:i]) for i in range(252, len(aux_ds.f_df))])
        aux_ds.f_df['halflife'] = np.zeros(len(aux_ds.f_df))
        aux_ds.f_df['halflife'][252:] = b[:,1]
        
    def filter_instrument_list (self):
        pass
    
    def updatePCA (self):
        LogManager.get_logger ().info ('Started updating PCA object')
        
        self.pca = PCA(from_time = str (get_from_time (self.last_timestamp, 
                                                self.rule.timeframe,
                                                days = 30)), #need enough data
                          to_time=self.last_timestamp, 
                          bLoadCandlesOnline=True, 
                          instrument_list=full_instrument_list)
                
        self.pca.load_input_data ()
        self.pca.extract_principal_components ()
        
        LogManager.get_logger ().info ('Finished updating PCA object')
    
    def updateSignals (self, last_timestamp, rule=None, 
                       instrument=None, instrument_list=None, 
                       bComputePC = True, bLoadDatasets = True):        
        
        self.reset_signals ()
        open_slots = self.get_open_slots ()
        self.pred_dict = {}
        
        if open_slots > 0:
# =============================================================================
#             dti = pd.to_datetime([last_timestamp])
#             ts = dt.datetime(dti.year, dti.month, dti.day, 
#                              dti.hour, dti.minute, dti.second)
#             from_time_pca = str(ts - relativedelta(days=30))
# =============================================================================
                        
            self.last_timestamp = last_timestamp
            
            try:
                self.pca
            except:
                bComputePC = True
                
            if bComputePC:
                #loads data for PCA
                self.updatePCA ()
            try:
                self.ds_holder
            except:
                bLoadDatasets = True
                
            if bLoadDatasets:
                other_ds = []
                for tf in self.rule.other_timeframes:
# =============================================================================
#                     if tf == 'D':
#                         from_time = str(ts - relativedelta(days=900))
#                     elif tf == 'H4':
#                         from_time = str(ts - relativedelta(days=200))
#                     elif tf == 'H1':
#                         from_time = str(ts - relativedelta(days=90))
#                     elif tf == 'M15':
#                         from_time = str(ts - relativedelta(days=40))
# =============================================================================
                        
                    other_ds.append(Dataset(timeframe=tf, 
                                            from_time = get_from_time(last_timestamp, tf), 
                                            to_time=self.last_timestamp,
                                            bLoadCandlesOnline = True).initOnlineConfig ())    
                    
# =============================================================================
#                 if self.rule.timeframe == 'D':
#                     from_time = str(ts - relativedelta(days=900))
#                 elif self.rule.timeframe == 'H4':
#                     from_time = str(ts - relativedelta(days=200))
#                 elif self.rule.timeframe == 'H1':
#                     from_time = str(ts - relativedelta(days=90))
#                 elif self.rule.timeframe == 'M15':
#                     from_time = str(ts - relativedelta(days=40))
# =============================================================================
                    
                ds = Dataset(timeframe=self.rule.timeframe, 
                                from_time = get_from_time(last_timestamp, self.rule.timeframe), 
                                to_time=self.last_timestamp,
                                bLoadCandlesOnline = True)            
                ds.initOnlineConfig ()
                
                self.other_ds = other_ds
                self.ds = ds
                
                if instrument_list is not None:
                    instrument_list = instrument_list
                elif instrument is not None:
                    instrument_list = [instrument]
                else:
                    instrument_list = self.instruments
            
            for ccy_pair in instrument_list:
                try:
                    if self.rule.filter_instruments (ccy_pair, self.last_timestamp):
                        if bLoadDatasets:
                            for ds in self.other_ds:
                                ds.loadSeriesOnline (instrument=ccy_pair)
                                ds.computeFeatures (bComputeHighLowFeatures=(self.rule.bUseHighLowFeatures if ds.timeframe == 'D' else False))
                            self.ds.loadSeriesOnline (instrument=ccy_pair)
                            self.ds.computeFeatures (bComputeHighLowFeatures=False)
                            
                            try:
                                if self.ds.df.index[-1] > self.pca.pca_df.index[-1]:
                                    self.updatePCA ()
                            except:
                                LogManager.get_logger ().error("Exception occurred", exc_info=True)
                            
                            self.pca.compute_pca_features(self.ds, bForceRecalc = True)
                            self.pca.save_pca_features (self.ds)
                            self.ds.loadPCAFeatures ()
            
                            self.ds_holder = DatasetHolder(instrument = ccy_pair,
                                                           from_time=self.ds.from_time, to_time=self.ds.to_time)
                            
                            self.ds_holder.ds_dict = {}
                            self.ds_holder.ds_dict[self.ds.ccy_pair+'_'+self.ds.timeframe] = self.ds
                            for ds in self.other_ds:
                                self.ds_holder.ds_dict[ds.ccy_pair+'_'+ds.timeframe] = ds
                            
                            self.compute_additional_features ()
                            
                        pred = self.rule.predict(self.ds_holder, verbose=True)
                        stop, target = self.rule.get_stop_target(self.ds.f_df)
                        self.ds.p_df = deepcopy(self.ds.f_df.ix[:, 0:6])
                        self.ds.p_df['Predictions'] = pred
                        self.pred_dict[ccy_pair] = self.ds.p_df['Predictions']
                        self.ds.removeSerialPredictions (self.serial_gap)
    
                        self.signals [ccy_pair] = {'signal':pred [-1] - NEUTRAL_SIGNAL, 
                                                    'stop': stop, 
                                                    'target' : target,
                                                    'last_px_in_dataset': self.ds.f_df['Close_' + self.ds.timeframe][-1]}
                        LogManager.get_logger ().info (self.name + ' - ' + ccy_pair + ' - ' + str (self.signals [ccy_pair]))
                        LogManager.get_logger ().info (str (self.ds.f_df.index[-1]) + ' : ' + str ([(col, self.ds.f_df.iat[-1, i]) for i, col in enumerate(self.ds.f_df.columns)]))
                        
                        
                    else:
                        #LogManager.get_logger ().info (self.name + ' - skipped calculating predictions for ' + ccy_pair)
                        self.signals [ccy_pair] = {'signal':NEUTRAL_SIGNAL, 
                                                    'stop': None, 
                                                    'target' : None,
                                                    'last_px_in_dataset': None}
                        
                except Exception as e:
                    LogManager.get_logger ().info ('Error updating prediction on: ' + ccy_pair + ' - ' + e.message)
                    LogManager.get_logger ().error("Exception occurred", exc_info=True)