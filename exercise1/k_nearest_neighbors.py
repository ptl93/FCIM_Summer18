#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tuan Le
@email: tuanle@hotmail.de

Implementation of k-nearest neighbours algorithm
"""

import numpy as np
from collections import Counter

class k_nearest_neighbors(object):
    
    ### Constructor ###
    def __init__(self, X, y, classif):
        '''
        Init method for K-Nearest-Neighbors algorithm.
        
        Parameters
        ----------
        X : Feature matrix
        y : Target vector
        classif : Boolean whether the algorithm should perform classification or regression.    
        Returns
        -------
        k_n_n object : an K-Nearest-Neighbours object for classification or regression
        '''
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.classif = classif
        if(classif):
            self.levels = np.unique(self.y)

    '''
    static method to compute euclidean distance
    '''
    @staticmethod
    def euclidean(xi, xj):
        di = np.sqrt(np.sum(np.square(xi - xj)))
        return di
    
    ### Train method ###
    def train(self, k = 5, X_data = None, y_data = None):
        ## Since prediction also calls train, we need to insert the *_data objects
        if X_data == None:
            X_data = self.X
        if y_data == None:
            y_data = self.y
        ## Init nearest_neighbors list
        self.nearest_neighbors = [None] * X_data.shape[0]
        ## Init pred vector for each observation
        self.y_pred = np.zeros(len(y_data))
        ## Now loop through each observation and compute the distance to all other points:
        for i in range(X_data.shape[0]):
            #for each iteration store distances from i to not_i in dist 
            dist = np.zeros(self.n)
            #dist = np.apply_along_axis(func1d=self.euclidean, axis=1, arr=X, xj = X[i:,])
            ## Inefficient way:
            for j in range(X_data.shape[0]):
                dist[j] = self.euclidean(xi = X_data[i,:], xj = X_data[j,:])
            ## Order distances and retrieve indexes
            #dist_ordered = np.sort(dist) ##for sorted
            ordered_idx = np.argsort(dist) ##for retrieving index of sorted array
            ## Take the k nearest neighbors
            k_nearest_neighbors = ordered_idx[0:k]
            ## Save results
            self.nearest_neighbors[i] = k_nearest_neighbors
            ## Predict
            if(self.classif):
                class_frequency_in_neighborhood = dict(Counter(y_data[k_nearest_neighbors]))
                ## apply majority vote, meaning getting the class which most neighbors have
                ## By default counter sorts in decreasing order, so we take the first key element from dict
                self.y_pred[i] = list(class_frequency_in_neighborhood.keys())[0]
            else:
                self.y_pred[i] = np.mean(a=y_data[k_nearest_neighbors])
        ## Now print performance measure:
        if(self.classif):
            acc = np.mean(y_data == self.y_pred)
            print("Accuracy in knn classification: {}".format(acc))
        else:
            mse = 1/len(y_data)* np.sum(np.square((y_data - self.y_pred)))
            print("Accuracy in knn classification: {}".format(mse))
        ##END train, return nothing
        return None
    
#    def predict(self, k = 5, X_test, y_test):
#        ## Call class train method
#        self.train(k, X_data = X_test, y_data = y_test)
#        return None
    
       
import sklearn.datasets as sdset
iris = sdset.load_iris()
X = iris["data"]
y = iris["target"]

my_knn = k_nearest_neighbors(X=X, y=y, classif=True)
my_knn.train(k=5)
#Accuracy in knn classification: 1.0