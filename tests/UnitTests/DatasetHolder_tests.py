# -*- coding: utf-8 -*-

from Framework.Dataset.DatasetHolder import *

dsh = DatasetHolder (from_time=2010, 
                     to_time=2011, 
                     instrument='EUR_USD')

dsh.loadMultiFrame(timeframe_list=['D', 'H4'])
dsh.appendTimeframesIntoOneDataset ()
