#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:57:14 2018

@author: joanna
"""

import os
import sys
from distutils.core import Extension, setup

os.chdir(os.path.dirname(os.path.abspath(__file__) ))
sys.argv = [sys.argv[0], 'build_ext', '-i']
setup(ext_modules = [Extension('_C_arraytest', ["C_arraytest.c"])])