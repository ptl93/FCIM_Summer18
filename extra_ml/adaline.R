### Implementation of ADALINE algorithm ###
### Tuan Le
### tuanle@hotmail.de

library(R6)

adaline = R6Class("adaline", public = list(
  X = NULL,
  y = NULL,
  n = NULL,
  p = NULL,
  w = NULL,
  loss_history = NULL,
  adaptive_lr = NULL,
  ### CONSTRUCTOR ###
  #' Initializes Adaline object
  #' @param X feature matrix
  #' @param y target vector
  #' @return Adaline object
  initialize = function(X, y) {
    self$X = X
    self$y = y
    self$n = nrow(X)
    self$p = ncol(X)
  },
  ### METHODS ###
  #' Computes L2 empirical risk
  #' @param X feature matrix
  #' @param y target vector
  #' @param w weights vector
  empirical_risk = function(X, y, w) {
    0.5 * sum((y - X %*% w)^2)
  },
  #' Computes weighted sum y_hat
  #' @param X feature matrix
  #' @param w weights vector
  weighted_sum = function(X, w) {
    X %*% w
  },
  #' Computes derivative of L2 loss with respect to weights
  #' @param y target vector
  #' @param y_hat predicted target vector
  #' @param X feature matrix
  l2_deriv = function(y, y_hat, X) {
    a = -(y - y_hat) 
    all_grad = apply(X, MARGIN = 2, function(x) x * a)
    return(colMeans(all_grad))
  },
  #' Trains the adaline object using either gradient or stochastic gradient descent. 
  #' The optimal learning rate will be computed using linesearch
  #' @param max_iter maximal number of gradient updates
  #' @optim optimization method. Either GD or SGD. Default is SGD
  #' @lr learning rate for weights update. Default is NULL. If NULL linesearch will be applied
  #' @batch_size batch_size for bootstrap sample in SGD
  train = function(max_iter, optim = "GD", lr = NULL, mini_batch = NULL) {
    ## Add intercept column in X matrix
    self$X = cbind(1, self$X)
    ## Init random weight vector
    self$w = rnorm(n = self$p + 1)
    ## Init loss_history
    self$loss_history = numeric(max_iter)
    for(i in seq.int(max_iter)) {
      y_hat = self$weighted_sum(self$X, self$w)
      ## Compute empirical risk
      self$loss_history[i] = self$empirical_risk(self$X, self$y, self$w)
      print(sprintf("Empirical risk in iteration %i: %f",i, round(self$loss_history[i], 2)))
      ## Do update : either gradient descent or stochastic gradient descent
      if(optim == "GD") {
        idx = 1:nrow(self$X)
      } 
      else if (optim == "SGD") {
        if(is.null(mini_batch)) mini_batch = ceiling(nrow(self$X) / 4)
        idx = sample(1:nrow(self$X), size = mini_batch, replace = TRUE)
      }
      ## Compute gradient from idx observations (either GD then whole dataset or SGD dann bootstrap minibatch)
      grad = self$l2_deriv(y = self$y[idx], y_hat = self$weighted_sum(self$X[idx,], self$w), X = self$X[idx, ])
      ## Compute optimal learning rate using line search
      if(is.null(lr)) {
        ## Enhance line search:
        obj = function(adapt_lr) {
          tmp_w = self$w - adapt_lr*sum(grad)
          my_obj = self$empirical_risk(X = self$X, y = self$y, w = tmp_w)
          return(my_obj)
        }
        lr = optimize(obj, interval = c(0, 10000))$minimum
      }
      ## Apply weight update
      self$w = self$w - lr*grad
    }
  }
))

## Test (stochastic) gradient descent:
# generate random data in which y is a noisy function of x, hence: y = 1x + 3 + rnorm(1000) where rnorm is N(0,1) distributed
set.seed(1)
x = matrix(runif(300, -5, 5), ncol = 1L)
y = x + 3 + rnorm(300)

# fit a linear model
res_lm <- lm( y ~ x )

# plot the data and the model
plot(x,y, col = rgb(0.2,0.4,0.6,0.4), main = "Linear regression: y = 1x + 3 + N(0,1)", type = "p")
abline(res_lm, col = "blue")

my_adaline = adaline$new(X = x, y = y)
my_adaline$train(max_iter = 200L, optim = "GD")
my_adaline$w
#[1] 3.0170338 0.9732514
print(res_lm$coefficients)
#(Intercept)           x 
#3.0170338   0.9732514 

## Use stochastic gradient descent
my_adaline2 = adaline$new(X = x, y = y)
my_adaline2$train(max_iter = 200L, optim = "SGD")
my_adaline2$w
#[1] 3.0054629 0.9220348

### Test multiple regression:

X = cbind(runif(100, -5, 5), rnorm(100, 10,2), rnorm(50, 20, 2))
weights = c(1.5, 10, -5)
y = 3 + X %*% weights + rnorm(100)

res_ml = lm(y ~ X)
print(res_ml$coefficients)
#(Intercept)          X1          X2          X3 
#0.7288864   1.5165656  10.1248789  -4.9441442 

my_adaline3 = adaline$new(X, y)
my_adaline3$train(max_iter = 2000L, optim = "GD")
print(my_adaline3$w)
#[1]  0.2913769  1.5164821 10.1227634 -4.9211366
