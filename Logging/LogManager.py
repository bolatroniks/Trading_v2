# -*- coding: utf-8 -*-

# logging_example.py

import os
import logging
from datetime import datetime as dt

from Config.const_and_paths import CONFIG_LOG_PATH

class LogManager ():
    #static attribute
    logger = None
    
    @staticmethod
    def initialize (name='default'):
        if name is None:
            name = 'default'
            
        if LogManager.logger is not None:
            return
        
        LogManager.logger = logging.getLogger(name)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        try:
            os.stat(os.path.join (CONFIG_LOG_PATH, name))
        except:
            os.mkdir(os.path.join (CONFIG_LOG_PATH, name))
            
        f_handler = logging.FileHandler(os.path.join (CONFIG_LOG_PATH, 
                                                      name, 'main_' + str(dt.now ())[:19].replace (':','-') + '.log'))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        LogManager.logger.addHandler(c_handler)
        LogManager.logger.addHandler(f_handler)

    @staticmethod
    def get_logger (name=None):
        if LogManager.logger is None:
            LogManager.initialize (name=name)
            LogManager.logger.setLevel (logging.INFO)
        return LogManager.logger
    