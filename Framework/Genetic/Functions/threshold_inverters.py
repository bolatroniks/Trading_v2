# -*- coding: utf-8 -*-

#ahuhauhau
def inv_fn_symmetric (x):
    return -x


def inv_fn_rsi (x):
    return 100 - x


def inv_fn_identity (x):
    return x

threshold_inverter_functions = {'symmetric': inv_fn_symmetric,
                                'RSI_like': inv_fn_rsi,
                                'identity': inv_fn_identity}