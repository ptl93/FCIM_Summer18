# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:01:32 2018

@author: tuan.le
"""

##Tensorflow intro

import tensorflow as tf 

#define variables and function
x1 = tf.Variable(3, name="x1")
x2 = tf.Variable(6, name="x2")
fnc = x1*2*x2 + x2

#open session to evaluate (graph):
session = tf.Session()
session.run(x1.initializer)
session.run(x2.initializer)
result = session.run(fnc)
print(result)
#42

#close session
session.close()

##another option:
with tf.Session() as session:
    x1.initializer.run()
    x2.initializer.run()
    result = fnc.eval()
    
##Use global variable initializer
init = tf.global_variables_initializer()

with tf.Session() as session:
    init.run()
    result = fnc.eval()

###Usage of tensorflow placeholders

y1 = tf.placeholder(tf.float32)
y2 = tf.placeholder(tf.float32)

sum_op = tf.add(y1,y2)
product_op = tf.multiply(y1,y2)

with tf.Session() as session:
    sum_result = session.run(sum_op, feed_dict={y1:36.0, y2:6.0})
    product_result = session.run(product_op, feed_dict={y1:6.0, y2:21.0})

print(sum_result, product_result)

#basic array arithmetic operations using tensorflow placeholders, feeding dictionaries with vectors
with tf.Session() as session:
    sum_result = session.run(sum_op, feed_dict={y1: [6.0,4.0,2.0], y2:[3.0,1.5,-10.2]})
    product_result = session.run(product_op, feed_dict={y1: [6.0,4.0,2.0], y2:[3.0,1.5,-10.2]})

print(sum_result)
print(product_result)

##Apply linear regression with tensorflow:

from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing.head()

#summary statistics
df_housing.describe()
#additional infos
df_housing.info()
##visualization:
import matplotlib.pyplot as plt
df_housing.hist(bins=50, figsize=(20,15))

#2D scatterplot of long. and lat. wrt to median income
df_housing.plot.scatter(x="Longitude", y="Latitude", alpha=0.4,
    s=df_housing["Population"]/100.0, label="population", figsize=(10,7),
    c="MedInc", 
    cmap=plt.get_cmap("jet"), 
    colorbar=True,
    sharex=False)

##Do linear regression using normal equations
import numpy as np

#get dimensions
n, p = df_housing.shape
#add bias
housing_df_bias = np.c_[np.ones((n,1)), housing.data]

X = tf.constant(housing_df_bias, dtype=tf.float32, name="X")
target = housing.target
target = target.reshape(-1,1) ##reshape array into vector
y = tf.constant(target, dtype=tf.float32, name="y")

#define normal equation and first create graph elements
X_transpose = tf.transpose(X)
# (X'X)^-1X'y
a = tf.matrix_inverse(tf.matmul(X_transpose, X))
b = tf.matmul(X_transpose, y)
theta = tf.matmul(a,b)

# create a session and evaluate theta
with tf.Session() as sess:
    theta_value = theta.eval()
    
print(theta_value)
