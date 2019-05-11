# -*- coding: utf-8 -*-

#hahaha
from copy import deepcopy

from Trading.Dataset.Dataset import Dataset

from Trading.Genetic.Functions.threshold_inverters import *
from Trading.Genetic.Functions.feature_functions import *
from Trading.Genetic.Functions.predictors import *

#some objects cannot be directly written to a file
#this function converts them to some basic info that
#can be stored

def prepare_dict_to_save (d_in):
    d = deepcopy (d_in)

    for k, v in d.iteritems ():
        if v.__class__.__name__ == 'dict':
            d[k] = prepare_dict_to_save (v)
        elif v.__class__.__name__ == 'function':
            d[k] = 'function:' + v.func_name
        elif v.__class__.__name__ == 'Dataset':
            d[k] = 'Dataset:' + v.timeframe

    return d

#does the opposite from the function above
#once a chromossome has been loaded from a file
#some objects are not yet ready to use
#this function converts them into something usable

def adapt_dict_loaded (d_in):
    d = deepcopy (d_in)

    for k, v in d.iteritems ():
        if v.__class__.__name__ == 'dict':
            d[k] = adapt_dict_loaded (v)
        if type (v) == str or type(v) == unicode:
            if v.find('function:') >= 0:
                d[k] = eval (v.replace ('function:', '').replace (' ', ''))
            elif v.find('Dataset:') >= 0:
                d[k] = Dataset (timeframe = v.replace ('Dataset:', '').replace (' ', ''))
    return d