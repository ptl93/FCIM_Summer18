# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:23:10 2018

@author: tuan.le
"""

import os
os.getcwd()
os.chdir("./desktop/sose18/FCIM_Summer18/exercise3")

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_features = iris.data
iris_target = iris.target

#Define Perceptron class:
class Softmax(object):
    """
    Softmax Classifier.
    
    Parameter
    ---------
    X : Feature Matrix (n,p) matrix
    y : Target vector 1n-array
    max_iter : maximal iteration
    
    Attributes
    ---------
    theta : 1p-array
        Weights after fitting.
    eta: float
        Learning rate
    max_iter: int
        maximal number of iterations
    ...
    """
    
    ### Constructor: ###
    def __init__(self, X, y, eta=0.1, add_intercept = True, max_iter=1000):
        '''
        Init method for Softmax.
        
        Parameter
        ---------
        X : Feature Matrix (n,p) matrix
        y : Target vector 1n-array
        max_iter : maximal iteration
            
        Returns
        -------
        Softmax object : a softmax object for classification
        '''
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1] - 1
        self.g = np.unique(y)
        self.theta = np.random.rand(d0=self.p, d1=self.g)
        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        self.max_iter = max_iter
        self.eta = eta
        