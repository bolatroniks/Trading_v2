#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 23:37:54 2017

@author: renato
"""
#%matplotlib inline
#Import libraries:
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from xgboost import plot_tree
from matplotlib.pylab import rcParams

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

from Trading.Dataset.Dataset import Dataset
from Trading.Training.TradingModel import TradingModel
import operator

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

global_c = 0.1
global_c2 = 0.75

def custom_obj(preds, dtrain):    
    labels = dtrain.get_label().astype(int)
    
    aux_labels = np.zeros((len(labels),3))
    
    for j in range(len(aux_labels)):
        aux_labels[j,labels[j]] = 1
   
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad_aux = np.zeros (np.shape(preds))
    hess_aux = np.zeros (np.shape(preds))
    
    for i in range(len (grad_aux)):
        if labels [i] == 0:
            grad_aux [i,0] = (preds[i,0]+preds[i,1]*global_c) - 1
            grad_aux [i,1] = ((preds[i,1]) - 0) * global_c2
            grad_aux [i,2] = (preds[i,2] - 0)
        elif labels [i] == 2:
            grad_aux [i,2] = (preds[i,2]+preds[i,1]*global_c) - 1
            grad_aux [i,1] = ((preds[i,1]) - 0) * global_c2
            grad_aux [i,0] = (preds[i,0] - 0)
        else:
            grad_aux [i,0] = (preds[i,0] - 0)
            grad_aux [i,2] = (preds[i,2] - 0)
            grad_aux [i,1] = (preds[i,1] - 1)
            
    
    #grad_aux [labels==0,:] = preds[labels==0,:] - aux_labels[labels==0,:]
    hess_aux = preds * (1.0-preds)
    
    #interleaves grad and hess
    grad = np.zeros(3*len(grad_aux))
    hess = np.zeros(3*len(hess_aux))
    for i in range(len (grad_aux)):
        for k in range (3):
            grad [i*3 + k] = grad_aux [i, k]
            hess [i*3 + k] = hess_aux [i, k]
    
    return grad, hess
    
def getSetsFromDataset (ds, bConvolveCdl=False, mm_200_idx=8, bNormalize_by_mm200=True, bShuffleTrainset = True):
    X = ds.X[:,-1,:]
    y = np.dot(ds.y, [0,1,2])
    cv_X = ds.cv_X[:,-1,:]
    cv_y = np.dot(ds.cv_y, [0,1,2])
    test_X = ds.test_X[:,-1,:]
    test_y = np.dot(ds.test_y, [0,1,2])
    
    if bConvolveCdl == True:
        exp_fn = np.exp(-np.linspace(0,40,20)/4)
        for i in range (81,142):
            #print i
            X[:,i] = np.convolve (X[:,i], exp_fn)[:len(X)]
            cv_X[:,i] = np.convolve (cv_X[:,i], exp_fn)[:len(cv_X)]
            test_X[:,i] = np.convolve (test_X[:,i], exp_fn)[:len(test_X)]
    
    if bNormalize_by_mm200 == True:
        for i in ds.mu_sigma_list:
            X[:,i] = X[:,i] / X[:,mm_200_idx]
            cv_X[:,i] = cv_X[:,i] / cv_X[:,mm_200_idx]
            test_X[:,i] = test_X[:,i] / test_X[:,mm_200_idx]

    #shuffle train set
    if bShuffleTrainset == True:
        idx = np.linspace(0,len(X)-1,len(X), dtype=int)
        np.random.shuffle(idx)
        X = X[idx, :]
        y = y[idx]

    return X, y, cv_X, cv_y, test_X, test_y


class TradingXGB (TradingModel):
    def __init__ (self, modelname, modelpath="./", total_no_series=120, dataset = None,
                  batch_size=128, lookback_window=252, n_features=75, no_signals=3, cv_set_size=1000, test_set_size=1000, pred_threshold=0.7,
                  isTraining=True, isTrainOnlyOnce=False, bZeroMA_train=False, bZeroMA_cv=False, bZeroMA_test=False,
                  featpath = './datasets/Fx/Featured/Normalized_complete',
                  parsedpath = './datasets/Fx/Parsed',
                  labelpath = './datasets/Fx/Labeled',
                  c1=0.1, c2=0.75, param = None, num_trees=20, bExpWeight=False):
        TradingModel.__init__ (self, modelname=modelname,
                               modelpath=modelpath, total_no_series=total_no_series, 
                               dataset=dataset, batch_size=batch_size,
                               lookback_window=2, n_features=142,
                               no_signals=no_signals, 
                               cv_set_size=cv_set_size, 
                               test_set_size=test_set_size, 
                               pred_threshold=pred_threshold,
                                  featpath=featpath,
                                  parsedpath=parsedpath,
                                  labelpath=labelpath)
        self.c1 = global_c = c1
        self.c2 = global_c2 = c2
        
        if param != None:
            self.param = param
        else:
            self.param = {}
            self.param = {}
            self.param['objective'] = 'multi:softprob'
            self.param['eval_metric'] = 'auc'
            self.param['max_depth'] = 6
            self.param['eta'] = 0.1
            self.param['subsample'] = 0.75
            self.param['colsample_bytree'] = 1.00
            self.param['silent'] = 0
            self.param['updater'] = 'grow_gpu'
            self.param['num_class'] = 3
            self.param['min_child_weight'] = 5
        self.num_trees = num_trees
        self.bExpWeight = bExpWeight
        self.feat_pts_dict = {}
        self.model = None
        
    def model_train (self, X, y, update=True):
        if self.bExpWeight == True:
            a_w = np.exp(np.linspace(0,len(X)-1, len(X))/len(X))
            a_w /= np.mean(a_w)
        else:
            a_w = np.ones(len(X))
    
        dtrain = xgb.DMatrix(X, label=y, weight=a_w)
        
        tmp = time.time ()
        if update == False or self.model is None:        
            bst = xgb.train(self.param, dtrain, self.num_trees, obj=custom_obj)
        elif self.model is not None:
            self.param.update({'process_type': 'update',
               'refresh_leaf': True})
            
            bst = xgb.train(self.param, dtrain, self.num_trees, obj=custom_obj, xgb_model=self.model)
            
        print ('Elapsed time in training: '+str(time.time () - tmp))
        self.model = bst
    
    def evaluate_model (self, cv_X, cv_y, bPrintCharts=True):
        preds = self.model.predict(xgb.DMatrix(cv_X))
        
        
        preds [:,0] = preds [:,0] > self.pred_threshold
        preds [:,2] = preds [:,2] > self.pred_threshold
        
        predictions = np.argmax(preds, axis=1)
        acc = metrics.accuracy_score(cv_y[predictions!=1], predictions[predictions!=1])    
        
        neutral = len(cv_y[predictions==1])
        
        feat_imp = pd.Series(self.model.get_fscore()).sort_values(ascending=False)
        if bPrintCharts == True:
            print "Accuracy : %.4g" % acc
            print ('Neutral predictions:'+str(neutral))
            plt.figure ()
            plt.hist (preds[:,0], bins=10, label='Short predictions')
            plt.legend(loc='best')
            plt.show ()                
            
            feat_imp[0:20].plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
        return acc, neutral, feat_imp
        
    def acc_feat_points (self, cv_X, cv_y):
        acc, neutral, feat_imp = self.evaluate_model (cv_X, cv_y, False)
        
        for i in range (20):
            idx = np.int(feat_imp.index[i].strip('f'))
            feat_name = self.dataset.feat_names_list [idx]
            
            if feat_name in self.feat_pts_dict.keys ():
                self.feat_pts_dict [feat_name] += feat_imp [i]
            else:
                self.feat_pts_dict [feat_name] = feat_imp [i]
 

series_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,40,41,42,43,44,46,47,48,49,50,
             51,52,53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 
             72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 98, 99,
             100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]

series_list2 = [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]

feat_sel = [75, 66, 13, 7, 76, 166, 78, 12, 11, 73, 182, 90, 0, 159, 6, 16, 60, 71, 
            158, 23, 156, 18, 15, 70, 64, 62, 5, 32, 10, 72, 83, 107, 51, 57, 3, 21, 
            89, 2, 46, 26, 61, 40, 69, 128, 56, 185, 117, 81, 157, 38]

if False:
    series_no = 1    
    
    ds = Dataset(lookback_window=2, n_features=142)
    ds.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume/NewFeatures'
    ds.feat_filename_prefix = 'not_normalized_new_feat_'
    #ds.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume'
    ds.period_ahead = 1
    ds.last=4000
    ds.cv_set_size = 250
    t_xgb = TradingXGB(modelname='trees', dataset=ds, num_trees=20)
    t_xgb.param['colsample_bytree'] = 0.5
    
    series_list = [1,3, 17]
    set_size_list = np.linspace(3000,2000,10)
    acc_list = []
    neutral_list = []
    
    for series_no in series_list:
        for test_set_size in set_size_list:
            ds.test_set_size = np.int(test_set_size)
        
            ds.loadSeriesByNo (series_no, bRelabel=False, bNormalize=False, bConvolveCdl=True)
            train_X, train_y, cv_X, cv_y, test_X, test_y = getSetsFromDataset (ds, bShuffleTrainset=True)
            #train_X = train_X[:, feat_sel]
            #cv_X = cv_X [:, feat_sel]
            
            t_xgb.model_train(train_X, train_y)
            #t_xgb.evaluate_model(a[-500:,:],b[-500:])
            t_xgb.acc_feat_points(cv_X, cv_y)        

    y = sorted(t_xgb.feat_pts_dict.items()[:], key=operator.itemgetter(1), reverse=True)
    
    n_feat = 80
    feat_sel = []
    feat_list = list(t_xgb.dataset.feat_names_list)
    for i in range (n_feat):
        feat_sel.append (feat_list.index(y[i][0]))

if True:
    feat_sel = [14,15,16,17,18,19,20,21,22,23,70,71,75,78,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167, 81, 84, 85, 86]
    #feat_sel = [14,15,16,17,18,19,20,21,22,23,70,71,75,78,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167]
    #feat_sel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
    series_no = 1
    param = {}
    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'auc'
    param['max_depth'] = 4
    param['eta'] = 0.1
    param['subsample'] = 0.75
    param['colsample_bytree'] = 1.0
    param['silent'] = 0
    param['updater'] = 'grow_gpu'
    param['num_class'] = 3
    param['min_child_weight'] = 5    
    
    ds = Dataset(lookback_window=2, n_features=142)
    ds.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume/NewFeatures'
    ds.feat_filename_prefix = 'not_normalized_new_feat_'
    #ds.featpath = './datasets/Fx/Featured/NotNormalizedNoVolume'
    ds.period_ahead = 1
    ds.last=4000
    ds.cv_set_size = 250
    t_xgb = TradingXGB(modelname='trees', dataset=ds, param=param, num_trees=20, c1=0.2, c2=0.9, pred_threshold=0.7)
    
    #t_xgb.model_train(train_X, train_y)
    
    set_size_list = np.linspace(2000,1000,5)
    
    acc_list = []
    neutral_list = []
    rcParams['figure.figsize'] = 30,18
    
    for i, test_set_size in enumerate(set_size_list[0:1]):
        ds.test_set_size = np.int(test_set_size)
    
        ds.loadSeriesByNo (series_no, bRelabel=False, bNormalize=False, bConvolveCdl=True)
        #ds.loadDataSet(series_list=[1,3], end=5, bRelabel=False, bNormalize=False)
        #ds.createSingleTrainSet(3)
        
        train_X, train_y, cv_X, cv_y, test_X, test_y = getSetsFromDataset (ds, bShuffleTrainset=True)
        train_X = train_X[:, feat_sel]
        print ('Length train_X: '+str(len(train_X)))
        cv_X = cv_X [:, feat_sel]        
        
        t_xgb.model_train(train_X, train_y)        

        acc, neutral, dummy = t_xgb.evaluate_model(train_X,train_y, bPrintCharts=False)
        print('acc and neutral on train set: '+str(acc)+', '+str(np.float(neutral)/np.float(len(train_X))))
        acc, neutral, dummy = t_xgb.evaluate_model(cv_X,cv_y, bPrintCharts=False)
        print('acc and neutral on cv set: '+str(acc)+', '+str(np.float(neutral)/np.float(len(cv_X))))
        
        acc_list.append(acc)
        neutral_list.append (neutral)
        #plot_tree(t_xgb.model, num_trees=0)
        
if False:
    series_no = 1
    set_size_list = np.linspace(2000,1000,9)
    
    acc_list = []
    neutral_list = []
    
    
    for i, test_set_size in enumerate(set_size_list):
        ds.test_set_size = np.int(test_set_size)
    
        ds.loadSeriesByNo (series_no, bRelabel=False, bNormalize=False, bConvolveCdl=True)
        train_X, train_y, cv_X, cv_y, test_X, test_y = getSetsFromDataset (ds, bShuffleTrainset=True)
        train_X = train_X[:, feat_sel]
        print ('Length train_X: '+str(len(train_X)))
        cv_X = cv_X [:, feat_sel]
        
        
        acc, neutral, dummy = t_xgb.evaluate_model(train_X,train_y, bPrintCharts=False)
        print('acc and neutral on train set: '+str(acc)+', '+str(np.float(neutral)/np.float(len(train_X))))
        acc, neutral, dummy = t_xgb.evaluate_model(cv_X,cv_y, bPrintCharts=False)
        print('acc and neutral on cv set: '+str(acc)+', '+str(np.float(neutral)/np.float(len(cv_X))))
        
        acc_list.append(acc)
        neutral_list.append (neutral)
        
        if acc <= 0.37 and np.float(neutral)/np.float(len(cv_X)) <= 0.9:
            t_xgb.model_train(train_X, train_y)
        