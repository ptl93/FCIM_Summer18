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
    ### CONSTRUCTOR ###
    #' Initializes Adaboost object
    #' @param data data.frame
    #' @param target
    #' @param add_intercept logical, whether intercept should be added. Default is FALSE
    #' @param formula formula for baselearner. Default is null. If null is inserted full model target ~ . will be taken
    #' @return Adaboost object
    initialize = function(data = NULL, target, add_intercept = FALSE, formula = NULL) {
      if (add_intercept) data = cbind(intercept = 1, data)
      self$X = data[, -which(names(data) == target)]
      self$y = data[[target]]
      if (length(levels(self$y)) > 2) stop("Target variable is not binary but multiclass")
      self$data = data
      self$n = nrow(data)
      self$p = ncol(data) - 1
      self$data = data
      self$classes = unique(self$y)
      self$base_models = list()
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
      self$beta_weights = numeric(length = max_iter)
      self$weights = list()
      library(rpart)
      #Equal weight init for each observation in trainig dataset
      self$weights = rep(x = 1/self$n, times = self$n)
      #Adaboost training loop
      for (iter in seq.int(max_iter)) {
        #Fit baselearner clasifier for training data with weights w_iter and get bHat_iter
        if (baselearner == "treestump") {
          #train tree stump
          self$base_models[[iter]] = rpart(formula = self$formula, data = self$data, weights = self$weights, 
            control = rpart.control(maxdepth = 1))
          self$base_preds[[iter]] = predict(self$base_models[[iter]], self$data, "class")
          #get indicator for missclassified observations
          indik_wrong = (self$base_preds[[iter]] != self$base_models[[iter]]$y)
          #compute weighted error for each observation
          self$error[[iter]] = sum(self$weights*indik_wrong) / sum(self$weights)
          #compute beta baselearner weights
          self$beta_weights[[iter]] = 0.5*log((1 - self$error[[iter]])/self$error[[iter]])
          #update weights
          w = w * exp(alpha[i] * missc)
          w = w / sum(w)
        }
      }
    },
    predict = function(newdata) {
  
    }
  )
)


data(BreastCancer, package = "mlbench")
data = BreastCancer
library(dplyr)
self = list()
myAdaboost = Adaboost$new(X, y, add_intercept = TRUE)
