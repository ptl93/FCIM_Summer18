## Tutorial 4 ##
## Tuan Le ##

##Exercise 3: Implementation of Adaboost for binary classification

library(R6)

#Create Adaboost Classifier Object:
Adaboost = R6Class("Adaboost",
  public = list(
    X = NULL,
    y = NULL,
    n = NULL,
    p = NULL,
    data = NULL,
    weights = NULL,
    classes = NULL,
    base_models = NULL,
    formula = NULL,
    beta_weights = NULL,
    base_preds = NULL,
    error = NULL,
    ### CONSTRUCTOR ###
    #' Initializes Adaboost object
    #' @param data data.frame
    #' @param target
    #' @param add_intercept logical, whether intercept should be added. Default is FALSE
    #' @param formula formula for baselearner. Default is null. If null is inserted full model target ~ . will be taken
    #' @return Adaboost object
    initialize = function(data, target, add_intercept = FALSE, formula = NULL) {
      if (add_intercept) data = cbind(intercept = 1, data)
      self$X = data[, -which(names(data) == target)]
      self$y = data[[target]]
      if (length(levels(self$y)) > 2) stop("Target variable is not binary but multiclass")
      self$data = data
      self$n = nrow(data)
      self$p = ncol(data) - 1
      self$data = data
      self$classes = unique(self$y)
      if (is.null(formula)) {
        self$formula = as.formula(paste(target, " ~ ."))
      } else {
        self$formula = formula
      }
    },
    ### METHODS ###
    #' Trains Adaboost binary classifier object
    #' @param baselearner weak baselearner used for Adaboost. Default is a tree strump with max.depth=1 from rpart
    #' @param max_iter maximal iteration for Adaboost
    #' @param ... additional parameters passed to baselearner algorithm
    train = function(baselearner = "treestrump", max_iter = 30L, ...) {
      self$base_models = vector("list", length = max_iter)
      self$base_preds = vector("list", length = max_iter)
      self$beta_weights = vector("list", length = max_iter)
      self$error = vector("list", length = max_iter)
      self$weights = vector("list", length = max_iter)
      library(rpart)
      #Equal weight init for each observation in trainig dataset
      self$weights[[1]] = rep(x = 1/self$n, times = self$n)
      #Adaboost training loop
      for (iter in seq.int(max_iter)) {
        #Fit baselearner clasifier for training data with weights w_iter and get bHat_iter
        if (baselearner == "treestump") {
          print(iter)
          #train tree stump
          ###Error in eval(extras, data, env) : object 'iter' not found ## In weights rpart call)
          self$base_models[[iter]] = rpart(formula = self$formula, data = self$data, weights = self$weights[[iter]], 
            control = rpart.control(maxdepth = 1))
          self$base_preds[[iter]] = as.vector(predict(self$base_models[[iter]], self$data))
          #compute weighted error
          self$error[[iter]] = self$get_error(y = self$base_models[[iter]]$y, y_hat = self$base_preds[[iter]], weights = self$weights[[iter]])
          #compute beta baselearner weights
          self$beta_weights[[iter]] = 0.5*log((1 - self$error[[iter]])/self$error[[iter]])
          #update weights
          self$weights[[iter + 1]] = self$weights[[iter]]*exp(self$beta_weights[[iter]]*self$get_missclassified_idx(y = self$base_models[[iter]]$y,
            y_hat = self$base_preds[[iter]]))
        }
      }
      return(NULL)
    },
    #' Get missclassified observation indexs
    #' @param y_hat predicted class from base learner
    #' @param y  true class label
    #' @return weighed_error
    get_missclassified_idx = function(y, y_hat) {
      indik_wrong = (y != y_hat)
    },
    #' Computes weighted error
    #' @param y_hat predicted class from base learner
    #' @param y  true class label
    #' @return weighed_error
    get_error = function(y, y_hat, weights) {
      indik_wrong = self$get_missclassified_idx(y, y_hat)
      weighted_error = sum(weights*indik_wrong) / sum(weights)
      return(weighted_error)
    },
    predict = function(newdata) {
  
    }
  )
)


data(BreastCancer, package = "mlbench")
data = BreastCancer
library(dplyr)

myAdaboost = Adaboost$new(data = data, target = "Class", add_intercept = FALSE)
myAdaboost$train(baselearner = "treestump", max_iter = 20L)
models = myAdaboost$base_models
