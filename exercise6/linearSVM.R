## Tutorial 6 ##
## Tuan Le ##

## Exercise 2: Implementation of linear soft-margin SupportVectorMachine (without intercept theta_0)
## Settings: 
# - can handle binary Classification task
# - use hinge-loss L(gamma) = max(0, 1-gamma) where gamma = y*x'theta

library(R6)
library(BBmisc)
linearSVM = R6Class("linearSVM",
  public = list(
    X = NULL,
    y = NULL,
    n = NULL,
    p = NULL,
    data = NULL,
    target = NULL,
    weights = NULL,
    max_iter = NULL,
    loss_history = NULL,
    optim_sol = NULL,
    ### CONSTRUCTOR ###
    #' Initializes linearSVM object
    #' @param data data.frame
    #' @param target
    #' @return linearSVM object
    initialize = function(data, target) {
      self$X = as.matrix(data[, -which(names(data) == target)])
      self$y = data[[target]]
      self$n = nrow(data)
      self$p = ncol(data) - 1
      self$weights = runif(n = self$p, min = -1, max = 1)
    },
    ### METHODS ###
    
    #' Compute hinge loss
    #' @param z vector
    #' @return hingeloss value for vector
    hinge_loss = function(z) {
      return(as.vector(pmax(0, 1 - z)))
    },
    total_loss = function(X, weights, y, C) {
      margins = y * X %*% weights
      empirical_risk = 1/self$n * sum( c(0.5*t(self$weights) %*% self$weights) + self$n*C*self$hinge_loss(margins) )
      return(empirical_risk)
    },
    deriv_hinge = function(x_i, y_i, weights) {
      dw = ifelse(y_i * x_i %*% weights < 1, -y_i*x_i, 0)
      return(dw)
    },
    subgradient_loss = function(x_i, weights, y_i, C)  {
      subgradient = weights + self$n * C * self$deriv_hinge(x_i = x_i, y_i = y_i, weights = weights)
      return(subgradient)
    },
    train = function(max_iter, C = 1, lr = NULL, optim_method = FALSE, b = 0, batch_size = NULL) {
      self$loss_history = numeric(max_iter)
      if (optim_method) {
        sum_loss = function(w, b = 0) { 
          z = (self$X %*% w + b) * self$Y
          L = sum(self$hinge_loss(z))
          objective_func = 0.5 * sum(w^2)
          return(objective_func + C * L)
        } 
        optim_func = function(x) sum_loss(w = self$weights)
        res = optim(par = self$weights, fn = optim_func, method = "Nelder-Mead",
          control = list(maxit = max_iter))
        par = res$par
        self$optim_sol = list(w = par, obj = res$val)
        print(self$optim_sol)
      } else {# Use stochstic sub gradient algorithm
        for (t in seq.int(max_iter)) {
          self$loss_history[t] = self$total_loss(X = self$X, weights = self$weights, y = self$y, C = C)
          messagef("Iteration: %i, Empirical risk: %f", t, self$loss_history[t])
          ## Enhance minibatch.  If no batch_size is inserted by default take bootstrap sample 25% of n_train
          if (is.null(batch_size)) batch_size = ceiling(0.25*self$n)
          mini_batch_idx = sample(x = 1:self$n, size = batch_size, replace = TRUE)
          ## Get subgradient:
          sub_grad = mean(self$subgradient_loss(x_i = self$X[mini_batch_idx,], weights = self$weights, y_i = self$y[mini_batch_idx], C = C))
          if (is.null(lr)) {
            ## Enhance line search:
            obj = function(adapt_lr) {
              tmp_weights = self$weights - adapt_lr*sum(self$subgradient_loss(x_i = self$X[mini_batch_idx,], weights = self$weights, y_i = self$y[mini_batch_idx], C = C))
              my_obj = self$total_loss(X = self$X, weights = tmp_weights, y = self$y, C)
              return(my_obj)
            }
            lr = optimize(obj, interval = c(0, 10000))$minimum
          }
          #adjust weights
          self$weights = self$weights - lr * sub_grad
        }
      }

      return(invisible(NULL))
    }
  )
)

set.seed(1337)
data = mlbench::mlbench.twonorm(n = 100, d = 2)
data = as.data.frame(data)
data$classes = sapply(1:nrow(data), function(i) {
  if (data[i,3] == 2) {
    -1
  } else {
    1
  }
})
target = "classes"

library(ggplot2)
fact_data = data
fact_data$classes = as.factor(fact_data$classes)
ggplot(fact_data, aes(x = x.1, y = x.2, color = classes)) + geom_point(shape = 1)

mylinSVM = linearSVM$new(data = data, target = target)
## Use stochastic sub gradient
mylinSVM$train(max_iter = 500, lr = NULL, batch_size = 20)
mylinSVM$weights
#[1] 0.6463704 1.4744407
## Get minimal empirical risk in loss_history:
min(mylinSVM$loss_history)
#[1] 11.02897
## Get iteration index for minimal empirical risk:
which.min(mylinSVM$loss_history)
#[1] 31

## Use nelder mead optim
mylinSVM2 = linearSVM$new(data = data, target = target)
mylinSVM2$train(max_iter = 5*nrow(data), optim_method = TRUE)
#$w
#[1] 0.7863688 0.9038573

#$obj
#[1] 0.7176669
