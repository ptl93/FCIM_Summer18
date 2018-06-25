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
      if (y_i * t(x_i) %*% weights < 1) {
        dw = -y_i*x_i
      } else {
        dw = 0
      }
      return(dw)
    },
    subgradient_loss = function(x_i, weights, y_i, C)  {
      subgradient = weights + self$n * C * self$deriv_hinge(x_i = x_i, y_i = y_i, weights = weights)
      return(subgradient)
    },
    train = function(max_iter, C = 1, lr = 0.01, optim_method = FALSE) {
      self$loss_history = numeric(max_iter)
      for (t in seq.int(max_iter)) {
        self$loss_history[t] = self$total_loss(X = self$X, weights = self$weights, y = self$y, C = C)
        messagef("Iteration: %i, Empirical risk: %f", t, self$loss_history[t])
        if (optim_method) {
          ## ToDO use optim method for parameter search
        } else {
          i = sample(seq.int(self$n), size = 1)
          self$weights = self$weights - lr*self$subgradient_loss(x_i = self$X[i,], weights = self$weights, y_i = self$y[i], C = C)
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
mylinSVM$train(max_iter = 5*nrow(data))
## somehow diverges bc stochastic subgradient?