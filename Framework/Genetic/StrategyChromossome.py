# -*- coding: utf-8 -*-

from Framework.Genetic.Chromossome import *
from Miscellaneous.Cache.CacheManager import CacheManager
from Config.const_and_paths import full_instrument_list


class StrategyChromossome ():
    def __init__ (self, **kwargs):
        crx = parse_kwargs (['crx', 'chromossome'], None, **kwargs)
        
        if crx is not None:
            self.crx = crx
            
        filename = parse_kwargs (['crx_file', 'crx_filename'], None, **kwargs)
        if filename is not None:        
            self.crx = Chromossome.load (filename)
            
        
            
kwargs2 = {
        'update_stop': {'func': fn_stop_update_trailing_v1, 
                        'kwargs': {
                                    'trailing_bars' : 10,
                                    'move_proportion' : 0.5,
                                }
                        },
        'init_target': {'func': fn_target_init_v1, 
                        'kwargs': {
                                    'trailing_bars' : 10,
                                    'target_multiple': 2,
                                }
                        },
        'init_stop': {'func': fn_stop_init_v1, 
                        'kwargs': {
                                    'trailing_bars' : 10,
                                    'move_proportion' : 0.5,
                                }
                        },
        'force_exit': {'func': fn_force_exit_n_bars, 
                        'kwargs': {
                                    'n_bars': 20,
                                }
                        }
        }