#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 00:47:51 2017

@author: renato
"""

import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def halflife(x):
    df = pd.core.frame.DataFrame(data=np.array(x), columns=['y'])
    df['ylag'] = df['y'].shift(1)
    df['deltaY'] = df['y'] - df['ylag']
    results = smf.ols('deltaY ~ ylag', data=df).fit()
    lam = results.params['ylag']
    halflife=-np.log(2)/lam
    return lam, halflife
    
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn

def hurst(ts):
    lags = range(2, 100)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)

	# Return the Hurst exponent from the polyfit output
    return poly[0]*2.0