# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 13:36:48 2016

@author: Joanna
"""

from nose.tools import *
from unittest import TestCase
import unittest
import numpy as np
from Trading.Training.TradingModel import TradingModel

test_mode = 'Prod'

class TestTradingModel (TestCase):
    def setUp (self):
        modelname = 'trained_model_normalized_feat_stateless_fixed_timesteps_more_complex_512_2x_more_dropout'
        modelpath = './models/weights'

        self.my_train = TradingModel (modelname=modelname,
                            modelpath=modelpath)
        
        print ("Setup!")
    
    def teardown (self):
        print ("Tear down!")
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadModel (self):        
        cfg = self.my_train.model.get_config ()
        assert (np.shape(cfg)[0]==5)
        print ("I ran!")
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadSeriesByNo (self):
                
        self.my_train.loadSeriesByNo(1)
        
        assert (self.my_train.dataset.X.shape[0]==4668)
        assert (self.my_train.dataset.cv_X.shape[0]==749)
        assert (self.my_train.dataset.test_X.shape[0]==749)
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadDataset (self):         
        self.my_train.loadDataSet(begin=1,end=4)
        
        assert (np.shape(self.my_train.dataset.dataset_list)==(3, 6))
        assert (len(self.my_train.dataset.dataset_list[0][0])==4882)
        self.my_train.evaluateOnLoadedDataset()
        self.my_train.createSingleTrainSet()
        assert (len(self.my_train.dataset.X)==14646)

    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadDatasetV2 (self):         
        self.my_train.loadDataSetV2(begin=1,end=4)
        
        assert (len(self.my_train.X)==14646)        