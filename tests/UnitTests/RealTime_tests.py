# -*- coding: utf-8 -*-


from unittest import TestCase
import unittest
from Framework.Dataset.RealTime import *

test_mode = 'Prod'

class TestRealTime (TestCase):
    def setUp (self):
        #self.my_ds = Dataset ()
        self.my_rt = RealTime ()
        
        print ("Setup!")
    
    def teardown (self):
        print ("Tear down!")
    
    @unittest.skipIf(False, "if to be implemented")
    def test_init (self):        
        self.my_rt = RealTime ()
        #assert (self.my_ds.featpath == './datasets/Fx/Featured/Normalized')
        print ("I ran!")
        
    @unittest.skipIf(False, "if to be implemented")
    def test_loadCandles (self):        
        candles = self.my_rt.loadCandlesToDataframe ()
        print_candles (candles)
        assert(1==0)
        #assert (self.my_ds.featpath == './datasets/Fx/Featured/Normalized')
        print ("I ran!")