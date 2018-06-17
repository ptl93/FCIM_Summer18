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

#Define Softmax class obect:
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
        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.g = len(np.unique(y))
        self.theta = np.random.rand(self.p, self.g)
        self.max_iter = max_iter
        self.eta = eta
    
    ### Methods: ###
    
    ###Softmax vectorized score###
    def softmax_vec(self, z):
        z = z - np.max(z)
        return np.exp(z)/np.sum(np.exp(z))
        
    def softmax_trafo(self, M):
        #init activated scores
        out_mat = np.zeros(shape=(M.shape[0], self.g)) 
        for i in np.arange(M.shape[0]):
            out_mat[i,:] = self.softmax_vec(M[i,:])
        return out_mat
        
    ###Weighted (linear) classifier sum###
    def weighted_sum(self, X, theta):
        return np.dot(X, theta)
        
    ###Computed posterior probabilites for each observation and class (vectorized)###
    def calc_posterior_prob(self, X, theta): 
        M = self.weighted_sum(X=X, theta=theta)
       # print("foo calc_posterior_prob")
       # print(M)
        posterior_probs = self.softmax_trafo(M=M)
        return posterior_probs 
    
    def one_hot_encoding(self, y):
        hot_encoded_mat = np.zeros(shape=(self.n, self.g))
        for i in np.arange(hot_encoded_mat.shape[0]):
            hot_encoded_mat[i, y[i]] = 1
        return hot_encoded_mat
    
    def get_objective(self, X, y, theta):
        probs_mat = self.calc_posterior_prob(X=X, theta=theta)
        prob_vals = np.zeros(shape=probs_mat.shape[0])
        for i in np.arange(probs_mat.shape[0]):
            prob_vals[i] = probs_mat[i, y[i]]
        return np.sum(np.log(prob_vals))
      
    def gradient(self, X, theta, onehot):
         probs = self.calc_posterior_prob(X=X, theta=theta)
         diff = onehot - probs
         grad = np.zeros(shape=(theta.shape[0], theta.shape[1]))
         for i in np.arange(X.shape[0]):
             xj = X[i,]  
             #fix bug for computing gradient matrix. [g,p] shape
             gradient = -np.outer(diff[i,:], xj)  
             grad = grad + gradient.T
         return grad
    
    def train(self, optimizer = "gd", eta = 0.05, max_iter = 200, batch_size = 10):
        self.eta = eta
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.objective = 10000
        if optimizer == "gd":
            #print("yes")
            onehot = self.one_hot_encoding(self.y)
            
            for i in np.arange(max_iter):
               # print(i)
                grad = self.gradient(X = self.X, theta = self.theta, onehot=onehot)
               # print(grad)
                #apply gradient update
                self.theta = self.theta - (self.eta) * grad 
                #print(self.theta)
                #compute objective value with updated theta weights
                self.objective = self.get_objective(X=self.X, y=self.y, theta=self.theta)
                print("Gradient Iteration: ", i)
                print("Objective Value: ", self.objective)
                #reference category is last col
                self.theta = self.theta - np.reshape(self.theta[:,(self.g-1)], newshape=(self.p,1))
            return {"theta": self.theta, "objective": self.objective}

## Conduct Softmax Regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size=0.33, random_state=42)

softmax = Softmax(X_train, y_train)
print(softmax.theta)

softmax.train()
#TypeError: weighted_sum() takes 2 positional arguments but 3 were given