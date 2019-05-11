# -*- coding: utf-8 -*-

class CacheManager ():
    cache = {}
    
    @staticmethod
    def get_cache ():
        return CacheManager.cache
    
    @staticmethod
    def clear_cache ():
        CacheManager.cache= {}

    @staticmethod
    def get_cached_object (obj):
        if obj in CacheManager.cache.keys ():
            return CacheManager.cache[obj]
        else:
            return None
    
    @staticmethod
    def cache_object (name, obj):
        CacheManager.cache [name] = obj