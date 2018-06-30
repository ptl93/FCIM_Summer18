### Percetron linear classifier ###
### Tuan Le
### tuanle@hotmail.de

library(R6)

Perceptron = R6Class("Perceptron", public = list(
  X = NULL,
  Y = NULL,
  N = NULL,
  M = NULL,
  weights = NULL,
  ### CONSTRUCTOR ###
  #' Initializes Perceptron object
  #' @param X feature matrix
  #' @param y target vector
  #' @return Perceptron object
  initialize = function(X, Y) {
    self$X = X
    self$Y = Y
    self$N = nrow(X)
    self$M = ncol(X)
  },
  ### METHODS ###
  #' Train Perceptron binary classifier object
  #' @param eta learning rate for weights adjusting
  #' @param max_iter maximal iteration for training
  train = function(eta = 0.1, max_iter = 1000L) {
    # Initiliaze weights
    self$weights = rnorm(n = self$M + 1)
    # Column bind bias
    self$X = cbind(1, self$X)
    # Compute Y_hat
    weighted_sum = self$X %*% self$weights
    Y_hat = sign(weighted_sum)
    for (i in seq.int(max_iter)) {
      if (all(self$Y == Y_hat)) {
        print(sprintf("Training stopped at iteration %i", i))
        break
      } else {
        for (sample in seq.int(self$N)) {
          if (self$Y[sample] != Y_hat[sample]) {
            # Apply weights update
            self$weights = c(self$weights + eta * self$Y[sample] * t(self$X[sample, ]))
            # Compute new Y_hat after one weight update
            weighted_sum = self$X %*% self$weights
            Y_hat = sign(weighted_sum)
          }
        }
      }
      if(i == max_iter) print(sprintf("Training stopped at maximal iteration %i", i))
    }
    print(paste("Final weights after training:", paste(self$weights, collapse = ",")))
    return(invisible(NULL))
  },
  visualize = function() {
    #TBD
    data_frame = cbind.data.frame(X = self$X, Y = as.factor(self$Y))
    colnames(data_frame) = c("bias", "X1", "X2", "Y")
    intercept = -self$weights[1]/self$weights[3] 
    slope = -self$weights[2]/self$weights[3]
    equation = paste0("X2 = ", round(slope, 2), "*X1 + ", round(intercept, 2))
    require(ggplot2)
    g_plot = ggplot(data = data_frame, mapping = aes(x = X1, y = X2, colour = Y)) + 
      geom_point(mapping = aes(x = X1, y = X2, fill = Y)) + geom_abline(slope = slope, intercept = intercept) +
      ggtitle(equation)
    return(g_plot)
  }
))

generate_data = function(n, dim) {
  ## create points in [-1,0]
  X1 = matrix(runif(ceiling(n/2)*dim), ncol = dim) - 1 
  ## create points in [0,1]
  X2 = matrix(runif(floor((n/2)*dim)), ncol = dim)
  ## row bind
  X = rbind(X1,X2)
  Y = c(rep(-1, ceiling(n/2)), rep(1, floor(n/2)))
  return(list(X = X, Y = Y))
}

set.seed(19)
data = generate_data(n = 20, dim = 2)
X = data$X
Y = data$Y

perceptron = Perceptron$new(X,Y)
perceptron$train()
plot = perceptron$visualize()
print(plot)
ggsave(filename = "perceptron_2D_R.png", plot = plot)
