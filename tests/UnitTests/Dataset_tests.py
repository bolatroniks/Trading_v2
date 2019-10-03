# -*- coding: utf-8 -*-

from Framework.Dataset.Dataset import Dataset
#from Framework.Training.TradingModel import TradingModel

from nose.tools import *
from unittest import TestCase
import unittest
import numpy as np

test_mode = 'Prod'

class TestDataset (TestCase):
    def setUp (self):
        #self.my_ds = Dataset ()
        self.my_ds = Dataset (ccy_pair='EUR_USD', timeframe='D',
                              from_time = 2010, to_time=2011)
        #modelname = 'trained_model_normalized_feat_stateless_fixed_timesteps_more_complex_512_2x_more_dropout'
        #modelpath = './models/weights'

        #self.my_train = TradingModel(modelname=modelname,
        #                             modelpath=modelpath)
        print ("Setup!")
    
    def teardown (self):
        print ("Tear down!")
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_init (self):        
        self.my_ds = Dataset ()
        assert (self.my_ds.featpath == './datasets/Oanda/Fx/Featured')
        print ("I ran!")
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadSeriesFromDisk (self):
        self.my_ds.loadCandles ()
        self.my_ds.loadFeatures ()
        self.my_ds.loadLabels ()
        
        assert(self.my_ds.df.shape == (713, 6))
        assert(len(self.my_ds.f_df) == 702)
        assert(len(self.my_ds.l_df) == 702)
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadSeriesOnline (self):
        self.my_ds.loadSeriesOnline ()
        assert (self.my_ds.df.shape == (715, 6))

#Testing loading macro data functionality
if False:
    ds = Dataset (ccy_pair='SPX500_USD', timeframe = 'H1', from_time = 2006, to_time = 2010)    
    ds.loadCandles ()
    ds.computeFeatures ()
    
    #loads macro dataframe
    df = ds.loadMacroData (filename = 'initial_claims.csv', bReturnMacroDf=True)    
    feat = 'ICSA'
    #computes an additional feature in the macro dataframe
    df['ICSA_Change'] = np.log(df[feat].rolling (window = 4).mean () / df[feat].shift (4).rolling (window = 12).mean ())
    
    #merges the macro dataframe into the features dataframe
    ds.mergeMacroDf(df)
    
#testing a gene based on macro data
if False:
    ds = Dataset (ccy_pair='SPX500_USD', timeframe = 'H1', from_time = 2006, to_time = 2010)