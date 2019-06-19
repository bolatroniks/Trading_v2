# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:29:35 2016

@author: Joanna
"""

#from dicttoxml import dicttoxml
from xml.etree import cElementTree as ET
from collections import defaultdict


def parse_kwargs (key, default_v, **kwargs):
    if type(key) == str:
        if key in kwargs.keys ():
            return kwargs[key]
        else:
            return default_v
        
    if type (key) == list:
        for _ in key:
            if _ in kwargs.keys ():
                return kwargs [_]
            
        return default_v

def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        print ('Children: ' + str(len(children)))
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            try:
                for k, v in dc.items():
                    print (str('dc.items(): ' + str(dc.items())))
                    dd[k].append(v)
            except:
                print ('error processing dc items: ' + str(dc))
        d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        print ('Text: ' + str(text))
        if children or t.attrib:
            if text:
              d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

def dict_to_etree(d):
    def _to_etree(d, root):
        if not d:
            pass
        elif isinstance(d, basestring):
            root.text = d
        elif isinstance(d, dict):
            for k,v in d.items():
                assert isinstance(k, basestring)
                if k.startswith('#'):
                    assert k == '#text' and isinstance(v, basestring)
                    root.text = v
                elif k.startswith('@'):
                    assert isinstance(v, basestring)
                    root.set(k[1:], v)
                elif isinstance(v, list):
                    for e in v:
                        _to_etree(e, ET.SubElement(root, k))
                else:
                    _to_etree(v, ET.SubElement(root, k))
        else:
            raise TypeError('invalid type: ' + str(type(d)))
    assert isinstance(d, dict) and len(d) == 1
    tag, body = next(iter(d.items()))
    node = ET.Element(tag)
    _to_etree(body, node)
    return ET.tostring(node)
 

def xmltodict (xml):
    return clean_dict(etree_to_dict(ET.XML(xml)))

#replaces dictionaries of the kinds of {'item': [a, b]} by [a, b]
def clean_dict (d):
    if type(d) != dict:
        return d
    for k, v in d.iteritems():
        if k == 'item':
            return v
        d[k] = clean_dict(v)
    
    return d

def p2f(x):
    return float(x.strip('%'))/100

#!/usr/bin/env python
try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import DateFormatter, WeekdayLocator,\
        DayLocator, MONDAY
    from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
except:
    pass

if False:
    default_periods_path = './'
    default_periods_filename = 'Periods of Interest.xml'

    default_periods = {'Periods': {'USD_ZAR_2014_2015_bull_market': {'from_time': '2014-07-01 00:00:00',
          'instrument_list': ['USD_ZAR'],
          'timeframes': ['D', 'H4'],
          'to_time': '2015-07-31 23:59:59'},
          
          'USD_bull_market': {'from_time': '2014-07-01 00:00:00',
          'instrument_list': ['USD_ZAR', 'USD_CAD', 'USD_NOK', 'USD_TRY', 'USD_JPY'],
          'timeframes': ['D', 'H4'],
          'to_time': '2015-07-31 23:59:59'},
          
          'JPY_Abenomics_deval': {'from_time': '2012-10-01 00:00:00',
          'instrument_list': ['USD_JPY'],
          'timeframes': ['D', 'H4'],
          'to_time': '2013-05-01 23:59:59'},
          
          'TRY_CBRT_deval': {'from_time': '2010-12-31 00:00:00',
          'instrument_list': ['USD_TRY'],
          'timeframes': ['D', 'H4'],
          'to_time': '2011-12-31 23:59:59'},
          }}
          
    xml = dict_to_etree(default_periods)
    
    test_dict = xmltodict(xml)

    