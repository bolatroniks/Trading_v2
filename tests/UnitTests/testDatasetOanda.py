# -*- coding: utf-8 -*-

from Trading.Dataset.Dataset import Dataset

from nose.tools import *
from unittest import TestCase
import unittest
import numpy as np

test_mode = 'Prod'

class TestDataset (TestCase):
    def setUp (self):        
        self.my_ds = Dataset(ccy_pair='USD_ZAR', timeframe='D', from_time=2001, to_time=2017)
        self.my_ds.initOnlineConfig ()
        print ("Setup!")
    
    def teardown (self):
        print ("Tear down!")
    
    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadSeriesByNo (self):
        self.my_ds.loadSeriesByNo(1)
        
        assert (self.my_ds.X.shape[0]==4668)
        assert (self.my_ds.cv_X.shape[0]==749)
        assert (self.my_ds.test_X.shape[0]==749)

    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_compute_return (self):
        self.my_ds.loadSeriesByNo(1)
        ret, long_hits, long_misses, short_hits, short_misses = self.my_ds.compute_return(self.my_train.model)
        print (np.sum(ret))
        assert(np.sum(ret)==132.0)

    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadDataset(self):
        self.my_ds.loadDataSet(begin=1, end=4)

        assert (np.shape(self.my_ds.dataset_list) == (3, 6))
        assert (len(self.my_ds.dataset_list[0][0]) == 4882)
        self.my_ds.evaluateOnLoadedDataset(self.my_train.model)
        self.my_ds.createSingleTrainSet()
        assert (len(self.my_ds.X) == 14646)

    @unittest.skipIf(test_mode == 'Dev', "if to be implemented")
    def test_loadDatasetV2(self):
        self.my_ds.loadDataSetV2(begin=1, end=4)
        print ('Shape my_ds.X: '+str(np.shape(self.my_ds.X)))
        assert (len(self.my_ds.X) == 14646)