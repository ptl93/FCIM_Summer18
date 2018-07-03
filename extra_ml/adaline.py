#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tuan Le
@email: tuanle@hotmail.de
"""

import numpy as np

class adaline(object):
    
    ## Constructor ##
    def __init__(self, X, y):
        '''
        Init method for Adaline Regression.
        
        Parameters
        ----------
        X : Feature matrix
        y : target vector
            
        Returns
        -------
        adaline object : an adaline object for regression
        '''
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1]
    
    ## Methods ##
    def empirical_risk(self, X, y, w):
        '''
        Computes L2 empirical risk
        
        Paramaters
        ----------
        X : Feature matrix
        y : target vector
        w : weight vector
        
        Returns
        -------
        Empirical Risk from the dataset
        '''
        residuals = np.square(y - np.matmul(X, w))
        empirical_risk = 0.5*np.sum(residuals)
        return empirical_risk
    
    def l2_deriv(self, y, y_hat, X):
        '''
        Computes the derivative of the L2 loss wrt. the weights
        
        Paramaters
        ----------
        y : true target vector
        y_hat : predicted target vector
        X : Feature matrix
        
        Returns
        -------
        (averaged) derivative of loss wrt. to weights
        '''
        a = -(y.reshape((self.n,1)) - y_hat.reshape((self.n,1)))
        all_grad = self.X * a
        averaged_grad = np.mean(all_grad, axis=0)
        return averaged_grad
    
    def train(self, max_iter, optim = "GD", mini_batch_size = None, lr = 0.01):
        '''
        Computes the derivative of the L2 loss wrt. the weights
        
        Paramaters
        ----------
        max_iter : Maximal number of iterations (for weight update)
        optim : Optimization method. Either Gradient Descent 'GD' or Stochastic Gradient Descent 'SGD' 
        mini_batch_size : size of mini batch for stochastic gradient descent
        lr : learning rate for weight update
        
        Returns
        -------
        None
        '''
        ## Concatenate intercept into feature matrix
        self.X = np.column_stack((np.ones(shape=(self.n,1)), self.X))
        ## Init random weights
        self.w = np.random.normal(size=self.p+1)
        ## Init loss history
        self.loss_history = np.zeros(max_iter)
        ## Train the adaline 
        for i in range(max_iter):
            y_hat = np.matmul(self.X, self.w)
            #y_hat = y_hat.reshape((self.n,1))
            self.loss_history[i] = self.empirical_risk(X=self.X, y=self.y, w=self.w)
            print("Empirical risk in iteration:" + str(i) + ": " + str(self.loss_history[i]))
            if optim == "GD":
                idx = np.arange(self.n)
            elif optim == "SGD":
                if mini_batch_size == None:
                    mini_batch_size = np.ceil(self.n/4)
                idx = np.random.choice(a=np.arange(self.n), size=mini_batch_size, replace=True)
            grad = self.l2_deriv(y=self.y[idx], y_hat=y_hat[idx], X=self.X[idx,:])
            self.w = self.w - lr*grad
            
            
### Testing adaline regression: ###
X = np.random.uniform(low=-5, high=5, size=(100,1))
weights = np.array([2])
intercept = 3
y = np.matmul(X, weights) + intercept + np.random.normal(size=100)
y = y.reshape((100,1))

my_adaline = adaline(X=X, y=y)
my_adaline.train(max_iter=2000, lr=0.01)
my_adaline.w
#array([2.97997842, 1.97478091])