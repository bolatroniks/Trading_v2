# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    #remove all neutral predictions from y_pred and y_true
    y_true = y_true[y_pred[:,1]!=True]
    y_pred = y_pred[y_pred[:,1]!=True]    

    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    #cce = T.nnet.kullback_leibler_divergence(y_pred, y_true)
    return cce


    #except:
    #    return 0, 0, 0