#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 08:50:22 2018

@author: tuanle
"""

###Author: Tuan Le
###Email: tuanle@hotmail.de

## Task:
## Implement quadratic discriminant analysis as explained in lecture


import pandas as pd
import numpy as np
from sklearn import datasets
from scipy.stats import multivariate_normal

iris = datasets.load_iris()
iris_features = pd.DataFrame(iris.data)
iris_features.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
iris_target = pd.DataFrame(iris.target)
iris_target.columns = ["Species"]
iris_target["Species"] = iris_target["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

iris_data = pd.concat([iris_features, iris_target], axis = 1)

class myQDA:
    #constructor
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.n = data.shape[0]
        self.p = data.shape[1] - 1
        self.classes = data[target].unique()
        self.n_j = data[target].value_counts()
        self.label_col = np.where(self.data.columns == self.target)
        self.pi_j = None
        self.mu_j = None
        self.sigma_j = None
        self.probs = None

    #helpers
    def dropCol(self, data, col):
        return data.drop(data.columns[[col]], axis = 1)
        

    def estimateParams(self):
        self.pi_j = self.n_j / self.n
        #calculate class means and covariance
        #initialize empty dictionaries
        self.mu_j = {cl: None for cl in self.classes} 
        self.sigma_j = {cl: None for cl in self.classes} 
        for cl in self.classes:
            #filter dataset by class
            idx = self.data[self.target] == cl
            filtered_data = self.data[idx].drop([self.target], axis=1)
            self.mu_j[cl] = filtered_data.mean()
            self.sigma_j[cl] = np.cov(filtered_data)
        return({'pi_j': self.pi_j, 'mu_j':self.mu_j ,'sigma_j':self.sigma_j })
    
    def predictQDA(self, data):
        data = data
        self.probs = pd.DataFrame(columns = self.classes)
        for cl in self.classes:
            self.probs[cl] = multivariate_normal(data, mean = self.mu_j[cl], cov = self.sigma_j[cl]) * self.pi_j[cl]
        return({"class_prob": self.probs})
        
        



     

        

        
testQDA = myQDA(data = iris_data, target = "Species")
params = testQDA.estimateParams()

#probs = testQDA.predictQDA(data = iris_data[:-1])

print(testQDA.pi_j)
print(testQDA.mu_j)
print(testQDA.sigma_j)
type(testQDA.mu_j)
