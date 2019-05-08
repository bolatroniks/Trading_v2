# -*- coding: utf-8 -*-



from nose.tools import *
from unittest import TestCase
import unittest
import numpy as np
from Trading.Dataset.RealTime import *

from Trading.Dataset.Oanda import *
from Config.const_and_paths import *

import sys

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