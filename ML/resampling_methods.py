# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:44:34 2018

@author: tuan.le
Resampling Methods - Machine Learning
"""

##task 1 - implement holdout strategy [train and test set]
from random import seed
from sklearn.datasets import load_iris
import numpy as np
from numpy.random import choice
import pandas as pd
from math import floor

# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
    '''
    Splitting dataset in train and test data.
        
    Parameters
    ----------
    dataset : the dataset to be splitted
    split : the ratio of data in the train data (default: 0.6)

    Returns
    -------
    train_data, test_data: datasets containing the train data, respectively, test data 
    '''
    n, p = dataset.shape
    n_train = floor(n*split)
    train_idx = choice(np.arange(n), replace=False, size=n_train)
    test_idx = np.setdiff1d(np.arange(n), train_idx)
    train_data = dataset.iloc[train_idx, :]
    test_data = dataset.iloc[test_idx, :]
    return train_data, test_data


#example: iris data
#http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
seed(42)
iris = load_iris()
X_data = iris.data
target_data = iris.target.reshape(-1,1)
iris_data = np.c_[X_data, target_data]
def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
colnames = [iris.feature_names, 'species']
colnames = flatten(colnames)
iris_data = pd.DataFrame(iris_data)
iris_data.columns = colnames

iris_train, iris_test = train_test_split(dataset = iris_data)


##task 2 - implement k-fold cross-validation
from random import seed
from sklearn.datasets import load_iris
import numpy as np
from numpy.random import choice
import pandas as pd
from math import floor

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    '''
    Generating a cross validation split of a datasets. 
    The function splits the data into k-folds with k 
    being the number of folds.
        
    Parameters
    ----------
    dataset : dataset which is splitted
    folds: number of folds (default: 3)

    Returns
    -------
    dataset_split: a list containing the separated folds
    '''
    n, p = dataset.shape
    #create empty dictionary which stores all folds
    k_folds = {fold: None for fold in map('fold{}'.format, range(1, folds+1))} 
    #compute split ratio for each cross-validation fold by ((k-1)/k * n) / n
    split_ratio =  ((folds-1)/folds * n)/n
    #for each fold call train_test_split function
    for fold in k_folds:
        train_data, test_data = train_test_split(dataset = dataset, split = split_ratio)
        k_folds[fold] = {"train":train_data, "test":test_data}
    return k_folds

iris_folds = cross_validation_split(iris_data, folds=4)
