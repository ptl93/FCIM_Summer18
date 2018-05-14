### Predictive Modelling, Exercise 3 
### Implement the softmax regression with gradient descent algorithm.
### Tuan Le
### tuanle@hotmail.de

###############################################################################################
###  R6 Implementation of SoftMax-Regression
###############################################################################################

library(R6)


# Create softmax_regression object

softmax_regression = R6Class("softmax_regression",
  public = list(
    ### Attributes ###
    X = NULL,
    y = NULL,
    n = NULL,
    p = NULL,
    g = NULL,
    theta = NULL,
    eta = NULL,
    optimizer = NULL,
    max_iter = NULL,
    objective = NULL,
    add_intercept = NULL,
    ### Constructor ###
    initialize = function(X, y, add_intercept = TRUE) {
      #' Initializes Softmax Regression Object
      #' @param X n x p matrix. n obs, p features. [if 1-vector is not included it will be added]
      #' @param y target vector of classification problem
      #' @param add_intercept logical whether to add an intercept or not
      #' @return The softmax_regression object with initialized values

      #Concatenate intercept if not existent
      bias_flag = TRUE
      for (col in seq.int(ncol(X))) {
        if (all(X[, col] == rep(1,nrow(X)))) bias_flag = FALSE
      }
      if (bias_flag && add_intercept) X = cbind(X, 1)
      #Fill in initial attributes
      self$X = X
      self$y = y
      self$n = nrow(X)
      self$p = ncol(X)
      self$g = length(unique(y))
      #Initialize theta class matrix randomly. Output: (p,g) matrix
      self$theta = matrix(runif(n = self$p*self$g, min = -1, max = 1), nrow = self$p, ncol = self$g)
    },
    ### METHODS ###
    
    #' Computes g softmax values for vector z 
    #' @param z g dimensional vector
    #' @return softmax values-vector
    softmax_vec = function(z) {
      return(exp(z) / sum(exp(z)))
    },
    
    #' Computes softmax values for rows in a matrix
    #' @param x matrix, e.g  (n,g)
    #' @return softmax values for each row 
    softmax_trafo = function(x) {
      return(t(apply(x, 1, self$softmax_vec)))
    },
    
    #' Computes weighted sum input X with weights theta class matrix, (n,p) x (p,g)
    #' @param X input data 
    #' @param theta weights / coefficients to learn
    #' @return weighted sum results for each class g, hence output matrix (n,g)
    weighted_sum = function(X, theta) {
      return(X %*% theta)
    },
    
    #' Computes poster probabilities of weighted sum for each class. Will squash into range [0,1]
    #' @param X input 
    #' @param theta weights / coefficients to learn
    #' @return Posterior probabilites for each class g, hence output matrix (n,g)
    calc_posterior_prob = function(X, theta) {
      posterior_probs = self$softmax_trafo(self$weighted_sum(X, theta))
      return(posterior_probs)
    },
    
    #' Get batches for stochastic gradient descent algorithms
    #' @param batch_size batch size to sample
    #' @return Posterior probabilites for each class g, hence output matrix (n,g)
    get_batches = function(batch_size) {
      inds = sample(seq_len(self$n))
      return(split(inds, ceiling(seq_along(inds) / batch_size)))
    },
    
    #' Creates one-hot encoding for target vector y.
    #' @param y target vector y. Either numeric or with "real factors"
    #' @return  (n,g) binary 0/1 matrix, each row as exactly one "1" entry
    one_hot_encoding = function(y) {
      d = data.frame(y = factor(as.integer(y), levels = 1:length(unique(y))))
      return(model.matrix(~y-1, data = d))
    },
    
    #' Computes objective value for softmax regression [negative loglikelihood as derived]
    #' @param X Input matrix (features)
    #' @param y Target vector 
    #' @param theta class coefficient matrix 
    #' @return scalar value for objective (negative sum loglikelihood)
    #' 
    get_objective = function(X, y, theta) {
      #calc class posteriors for each observation
      probs_mat = self$calc_posterior_prob(X, theta)
      #get element per row of true class
      prob_vals = vector("numeric", length = nrow(probs_mat))
      for (row in seq.int(nrow(probs_mat))) {
        prob_vals[row] = probs_mat[row, y[row]]
      }
      return(-sum(log(prob_vals)))
    },
    
    #' Computes the gradient of the objective [negative loglikelihood]
    #' @param X Input matrix (features)
    #' @param y Target vector 
    #' @param theta class coefficient matrix 
    #' @return scalar value for objective (negative sum loglikelihood)
    # computes gradient of objective (neg lok lik)
    # as theta is a matrix also returns a matrix of same dim
    gradient = function(X, theta, onehot) {
      #calc posterior class prob for each observations (n,g)
      probs = self$calc_posterior_prob(X, theta)
      diff = onehot - probs  # (n,g)
      grad = matrix(0, nrow = nrow(theta), ncol = ncol(theta))
      # (p,g)
      
      for (i in 1:nrow(X)) { # iterate over all obs
        xj = X[i,]  #(1,g)
        gradient = -tcrossprod(diff[i,], xj)  # (1,g) and (p,1) outter product g*p
        grad = grad + t(gradient)
      }
      return(grad)
    },
    
    #' Trains the softmax regression problem using gradient descent
    #' @param optimizer optimization algorithm. Either gradientdescent (gd) or  stochastic gd (sdg)
    #' @param eta learning rate (stepsize) for optimization algorithm 
    #' @param max_iter maximal number of iteration
    #' @param batch_size batch_size for stochastic gradient descent
    #' @return List with optimal theta coefficients for each class and objective value
    # computes gradient of objective (neg log lik)
    # as theta is a matrix also returns a matrix of same dim
    train = function(optimizer = "gd", eta = 0.05, max_iter = 200L, batch_size = 10) {
      self$eta = eta
      self$optimizer = optimizer
      self$max_iter = max_iter
      self$objective = 10000
      #normal gradient descent
      if (self$optimizer == "gd") {
        #get one hot encoded matrix
        onehot = self$one_hot_encoding(self$y)
        #iterate over training sample
        for (iter in seq.int(self$max_iter)) {
          #compute gradient for all samples in train data
          grad = self$gradient(self$X, self$theta, onehot)
          #apply gradient update
          self$theta = self$theta - (self$eta) * grad 
          #compute objective value with updated theta weights
          self$objective = self$get_objective(self$X, self$y, self$theta)
          print(paste0("Gradient Iteration: ", iter))
          print(paste0("Objective Value: ", self$objective))
          self$theta = self$theta - self$theta[, self$g]
        }
        return(list(theta = self$theta, objective = self$objective))
      } else if (self$optimizer == "sgd") {
        for (e in seq_len(self$max_iter)) { # Start at epoch 1
          batches = self$get_batches(batch_size = batch_size)
          for (batch in batches) { # Iterate over Batches
          }
        }
      } else {
        stop("Either choose gd or sgd!")
      }
    },
    
    #' Gets column idx for maximal value in one row
    #' @param probs_mat (n,g) matrix containins class-probs for each observations
    #' @return numeric vector containins column idx (class)
    get_class_idx = function(probs_mat) {
     classes = apply(probs_mat, MARGIN = 1, function(row) {
       which.max(row)
     })
     return(classes)
    },
    
    #' Predicts softmax regression on (new) data
    #' @param X New (test) data matrix
    #' @param y target Vector of (test) data 
    #' @return List with optimal theta coefficients for each class and objective value
    # computes gradient of objective (neg log lik)
    # as theta is a matrix also returns a matrix of same dim
    predict = function(X, y = NULL) {
      bias_flag = TRUE
      for (col in seq.int(ncol(X))) {
        if (all(X[, col] == rep(1,nrow(X)))) bias_flag = FALSE
      }
      if (bias_flag) X = cbind(X, 1)
      #calc posteriorior class probabilities
      probs = self$calc_posterior_prob(X, self$theta)
      #get classes by "majority"/"maximal" rule
      classes = self$get_class_idx(probs)
      #compute test error if y is passed
      mmce = NULL
      if (!is.null(y))
        mmce = mean(y != classes)
      return(list(classes = classes, probs = probs, mmce = mmce))
    }
  )
)

## Conduct Softmax regression

X = iris[, -5]
y = as.integer(iris[, 5])

set.seed(12)
ind = sample(nrow(iris))
train_idx =  ind[1:100]
test_idx =  ind[101:150]
X_train = as.matrix(X[train_idx, ])
y_train = y[train_idx]

X_test = as.matrix(X[test_idx, ])
y_test = y[test_idx]

## Create softmax instance
softmax = softmax_regression$new(X_train, y_train)
print(softmax)
head(softmax$X)
softmax$theta

softmax$train(eta = 0.05, max_iter = 200L)
test_pred = softmax$predict(X_test, y_test)
print(test_pred)
#...
#$mmce
#[1] 0.04

### Compare to mlr:
library(mlr)
set.seed(123)
iris.task
lrn = makeLearner("classif.multinom")
holdout(learner = lrn, task = iris.task, split = 2/3, measures = mmce)

### Apply Softmax Regression with same data in own version
my_iris_task = makeClassifTask(id = "iris", data = iris, target = "Species")
mod = train(lrn, my_iris_task, subset = train_idx)
test_preds = predict(mod, my_iris_task, subset = test_idx)
performance(test_preds)
#mmce 
#0.02
