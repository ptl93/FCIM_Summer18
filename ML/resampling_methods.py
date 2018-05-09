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

print(iris_folds["fold1"]["train"].shape)
print(iris_folds["fold1"]["train"].head())
print(iris_folds["fold1"]["test"].shape)
print(iris_folds["fold1"]["test"].head())


print(iris_folds["fold2"]["train"].shape)
print(iris_folds["fold2"]["train"].head())
print(iris_folds["fold2"]["test"].shape)
print(iris_folds["fold2"]["test"].head())


## Learning Curves Task

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
%matplotlib inline

# number of instances
m = 100
# dataset
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
# plot
plt.figure()
plt.scatter(np.sort(X), np.sort(y))


def plot_learning_curve(model,X,y, test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]):
    '''
    Generating a plot of a learning curve
        
    Parameters
    ----------
    model : model which is used for training the dataset
    X: dataset
    y: labels     
    '''
    no_iter = len(test_sizes)
    performance_measure =  [None] * no_iter
    i = 0
    for size in test_sizes:
        # Create machine learning model
        ml_model = model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= size, random_state=42)

        # Train the model using the training sets
        ml_model.fit(X_train, y_train)
        
        # Make predictions using the testing set
        y_pred = ml_model.predict(X_test)
        performance_measure[i] = mean_squared_error(y_test, y_pred)
        i += 1
    plt.scatter(x=test_sizes, y=performance_measure)
    


from sklearn.linear_model import LinearRegression
ml_model = LinearRegression()
plot_learning_curve(ml_model, X,y)
