# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:48:25 2018

@author: tuan.le
"""

### Implementation of resampling methods

##import needed libraries


from sklearn.datasets import load_iris
import numpy as np
from numpy.random import choice
import pandas as pd
from math import ceil


def holdout_sampling(X, y, train_size = 2/3):
    '''
        Hold-Out-Sampling.
        
        Parameter
        ---------
        X : Feature Matrix (n,p) [numpy.ndarray]
        y : Target vector 1n-array [numpy.ndarray]
        train_size : Fraction of train size. Train dataset will contain ceil(train_size*X.shape[1]) observations
            
        Returns
        -------
       dictionary {train = [X_train, y_train], test = [X_test, y_test]}
    '''
    n,p = X.shape
    n_train = ceil(n*train_size)
    train_idx = choice(np.arange(n), replace=False, size=n_train)
    test_idx = np.setdiff1d(np.arange(n), train_idx)
    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]
    
    return {'train': [X_train, y_train], 'test': [X_test, y_test]}

def k_fold_cv(X, y, k = 5):
    '''
        K-Fold-Cross-Validation.
        
        Parameter
        ---------
        X : Feature Matrix (n,p) [numpy.ndarray]
        y : Target vector 1n-array [numpy.ndarray]
        k : number of cross validation iterations. Default is k = 5
            
        Returns
        -------
       nested dictionary with cv-iteration folds
    '''
    n,p = X.shape
    sampled_idx = choice(np.arange(n), size=n, replace=False)
    ## TODO use split idea https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html
    #Init empty dictionary with named keys via list comprehension
    k_fold_out = dict.fromkeys([i for i in range(1,k+1)])
    for key in k_fold_out.keys():
        k_fold_out[key] = 'foo as of now'
        
    return k_fold_out

def bootstrap_sampling(X, y, B = 50):
    '''
        K-Fold-Cross-Validation.
        
        Parameter
        ---------
        X : Feature Matrix (n,p) [numpy.ndarray]
        y : Target vector 1n-array [numpy.ndarray]
        B : Number of Bootstrap iterations. Default is B = 50
            
        Returns
        -------
       nested dictionary with Bootstrap Iterations
    '''
    n,p = X.shape
    #Init empty dictionary with named keys via list comprehension
    bootstrap_out = dict.fromkeys([i for i in range(1,B+1)])
    for key in bootstrap_out.keys():
        train_idx = choice(np.arange(n), replace=True, size=n)
        test_idx = np.setdiff1d(np.arange(n), train_idx)
        bootstrap_out[key] = {'train': [X[train_idx,], y[train_idx]], 'test': [X[test_idx,], y[test_idx]]}   
    return bootstrap_out

def subsampling(X, y, B = 50, train_size = 0.8):
    '''
        K-Fold-Cross-Validation.
        
        Parameter
        ---------
        X : Feature Matrix (n,p) [numpy.ndarray]
        y : Target vector 1n-array [numpy.ndarray]
        B : Number of Bootstrap iterations. Default is B = 50
        train_size: Fraction of training size in each subsampling iteration. Default is 0.8    
        Returns
        -------
       nested dictionary with Subsampling Iterations
    '''
    n,p = X.shape
    #Init empty dictionary with named keys via list comprehension
    subsampling_out = dict.fromkeys([i for i in range(1,B+1)])
    for key in subsampling_out.keys():
        subsampling_out[key] = holdout_sampling(X,y, train_size)
    return subsampling_out


iris = load_iris()
X = iris.data
y = iris.target
iris_holdout = holdout_sampling(X,y)
iris_cv = k_fold_cv(X,y)
