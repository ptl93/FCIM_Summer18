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

##clear whole workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

##Apply linear regression with tensorflow:

from sklearn.datasets import fetch_california_housing
import pandas as pd
import tensorflow as tf
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

##Compute theta_value by applying gradient descent algorithm
#import libraries for standardization: https://www.quora.com/Why-does-mean-normalization-help-in-gradient-descent
from sklearn.preprocessing import StandardScaler
#create scaler instance
scaler = StandardScaler()
#apply scaler on housing instance
scaled_housing_data = scaler.fit_transform(X=housing.data)
#add intercept
scaled_housing_data_bias = np.c_[np.ones((n,1)), scaled_housing_data]

##for gradient descent we need a learning rate and number of iterations / gradient steps to apply
n_epochs = 2000 #wrt to deeplearning context
learning_rate = 0.02

#Now in tensorflow create the matrices for the computational graph:
X = tf.constant(scaled_housing_data_bias, dtype=tf.float32, name="X")
y = tf.constant(target, dtype=tf.float32, name="y")

#for the regression (p+1) - parameters we randomly initialize them to be between -1 and 1 as starting values
theta = tf.Variable(tf.random_uniform(shape=[p + 1, 1], minval=-1, maxval=1, seed=85), name="theta")
#define computation for the y-pred value
y_pred = tf.matmul(X, theta, name="y_hat")
#define error/loss
error = y_pred - y
#now define loss function which should be minimized:
mse = tf.reduce_mean(tf.square(error), name="mse")
mae = tf.reduce_mean(tf.abs(error), name="mae")

#for mse the gradient (derivative) is: [note that we take the negative gradient bc of loss MINIMIZATION]
#if it were to be loss maximization we'd have the derivative -2/n * X'[Y-X*theta]
l2_gradients = 2/n * tf.matmul(tf.transpose(X), error)

#define the learning operation:
learning_op = tf.assign(theta, theta - learning_rate*l2_gradients)
#define global tf initiliazer
init = tf.global_variables_initializer()
#create session and evaluate the graph
with tf.Session() as session:
    session.run(init)
    #apply learning through epochs (full batch size)
    for epoch in range(n_epochs):
        #print every 100th epochs
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        #run the weights update
        session.run(learning_op)
    #save the best theta weight
    best_theta = theta.eval()

##Try automatic differentiation. Advantage--> gradient does not have to be defined as in code line 148
help(tf.gradients)
automatic_l2_gradients = tf.gradients(ys=mse, xs=[theta])[0]
#define the learning operation:
learning_op2 = tf.assign(theta, theta - learning_rate*automatic_l2_gradients)

with tf.Session() as session:
    session.run(init)
    #apply learning through epochs (full batch size)
    for epoch in range(n_epochs):
        #print every 100th epochs
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        #run the weights update
        session.run(learning_op2)
    #save the best theta weight
    best_theta2 = theta.eval()

#compare both thetas
print(best_theta)
print(best_theta2)

##use Mean Absolute deviation using automatic differentiation
automatic_l1_gradients = tf.gradients(ys=mae, xs=[theta])[0]
#define the learning operation:
learning_l1_op = tf.assign(theta, theta - learning_rate*automatic_l1_gradients)

with tf.Session() as session:
    session.run(init)
    #apply learning through epochs (full batch size)
    for epoch in range(n_epochs):
        #print every 100th epochs
        if epoch % 100 == 0:
            print("Epoch", epoch, "MAE =", mae.eval())
        #run the weights update
        session.run(learning_l1_op)
    #save the best theta weight
    best_theta3 = theta.eval()

print(best_theta)
print(best_theta3)
##output differs as expected


##Another way is to use tensorflow GradientDescentOptimizer
#get instance of the GradientDescentOptimizer and set the learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# define the training operation
training_op_l1 = optimizer.minimize(mae)
training_op_l2 = optimizer.minimize(mse)
init = tf.global_variables_initializer()
# create a session and evaluate the training_operation
with tf.Session() as session:
    session.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        session.run(training_op_l2)
        
    best_theta_l2 = theta.eval()


with tf.Session() as session:
    session.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MAE =", mae.eval())
        session.run(training_op_l1)
        
    
    best_theta_l1 = theta.eval()

print(best_theta_l2)
print(best_theta_l1)
