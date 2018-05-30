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
    max_iter = NULL,
    add_intercept = FALSE,
    ### CONSTRUCTOR ###
    #' Initializes Adaboost object
    #' @param data data.frame
    #' @param target
    #' @param add_intercept logical, whether intercept should be added. Default is FALSE
    #' @param formula formula for baselearner. Default is null. If null is inserted full model target ~ . will be taken
    #' @return Adaboost object
    initialize = function(data, target, add_intercept = FALSE, formula = NULL) {
      if (add_intercept) {
        data = cbind(intercept = 1, data)
        self$add_intercept = TRUE
      } 
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
      self$beta_weights = numeric(length = max_iter)
      self$error = vector("list", length = max_iter)
      #self$weights = vector("list", length = max_iter)
      library(rpart)
      #Equal weight init for each observation in trainig dataset
      self$weights = rep(x = 1/self$n, times = self$n)
      #Adaboost training loop
      for (iter in seq.int(max_iter)) {
        #Fit baselearner clasifier for training data with weights w_iter and get bHat_iter
        if (baselearner == "treestump") {
          #train tree stump
          self$base_models[[iter]] = rpart(formula = self$formula, data = self$data, weights =  self$weights, 
            control = rpart.control(maxdepth = 1))
          #http://stat.ethz.ch/R-manual/R-devel/library/rpart/html/predict.rpart.html , somehow "class" is not working
          self$base_preds[[iter]] = predict(object = self$base_models[[iter]], newdata = self$data, type = "vector")
          #print(paste("Iteration no. ", iter))
          #print(paste("Misclassification rate", sum(self$get_missclassified_idx(y = self$base_models[[iter]]$y, y_hat = self$base_preds[[iter]]))/self$n))
          #compute weighted error
          self$error[[iter]] = self$get_error(y = self$base_models[[iter]]$y, y_hat = self$base_preds[[iter]], weights = self$weights)
          #compute beta baselearner weights
          self$beta_weights[iter] = 0.5*log((1 - self$error[[iter]])/self$error[[iter]])
          #update weights
          self$weights = self$weights*exp(self$beta_weights[[iter]]*self$get_missclassified_idx(y = self$base_models[[iter]]$y,
            y_hat = self$base_preds[[iter]]))
          
          #normalize weights
          self$weights = self$weights / sum(self$weights)
        }
      }
      #save max_iter in object for prediction
      self$max_iter = max_iter
      return(invisible(NULL))
    },
    #' Get missclassified observation indexs
    #' @param y_hat predicted class from base learner
    #' @param y  true class label
    #' @return indikator for missclassified observations
    get_missclassified_idx = function(y, y_hat) {
      indik_wrong = (y != y_hat)
      return(indik_wrong)
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
    #' Predicts Adabost classifier on (new) data
    #' @param newdata newdata
    #' @return predicted class
    predict = function(newdata) {
      if (self$add_intercept) newdata = cbind(intercept = 1, newdata)
      #init prediction matrix for n observations and M base learners
      preds_mat = matrix(nrow = nrow(newdata), ncol = self$max_iter)
      for (model in seq.int(self$max_iter)) {
        preds_mat[, model] = predict(self$base_models[[model]], newdata, type = "vector") #returns label 1,2
      }
      #recode class to -1 and 1 for sign function. 1 to -1 and 2 to 1
      #print("Foo")
      #print(head(preds_mat))
      preds_mat = ifelse(preds_mat == 1, -1, 1)
      #print("Should be converted")
      #print(head(preds_mat))
      ##for each row observation now compute weigted average from base learners
      for (obs in seq.int(nrow(newdata))) {
        preds_mat[obs, ] = preds_mat[obs, ] * self$beta_weights
      }
      #print(head(preds_mat))
      ##compute sign of weighted average and take sign function
      preds = sign(rowSums(preds_mat))
      #print(preds)
      ##to compare to true value recode back to 1 and 2
      preds = ifelse(preds == -1, 1, 2)
      #print(preds)
      #compute mean missclassification rate
      mmce = sum((preds == as.numeric(self$y)))/nrow(newdata)
      list(preds = preds, mmce = mmce)
    }
  )
)


data(BreastCancer, package = "mlbench")
library(dplyr)
myAdaboost = Adaboost$new(data = BreastCancer[,-1], target = "Class", add_intercept = TRUE)
myAdaboost$train(baselearner = "treestump", max_iter = 50L)
myAdaboost$beta_weights
preds = myAdaboost$predict(BreastCancer)
preds$mmce
#[1] 0.9685265
(nrow(BreastCancer))
#699, overfitting. Also maybe because of max_iteration 50

##compare with ada function from ada package
library(ada)
#BreastCancer = cbind(intercept = 1, BreastCancer)
?ada
X_BreastCancer = BreastCancer %>% select(-Class, -Id)
X_BreastCancer = as.matrix(cbind(intercept = 1, X_BreastCancer))
y_BreastCancer = as.numeric(unlist(as.vector(BreastCancer %>% select(Class))))
ada_compared1 = ada(x = X_BreastCancer, y = y_BreastCancer, iter = 50L)
print(ada_compared1) 
#Train Error: 0.011 , better than own version

data(Ionosphere, package = "mlbench")

myAdaboost2 = Adaboost$new(data = Ionosphere, target = "Class", add_intercept = TRUE)
myAdaboost2$train(baselearner = "treestump", max_iter = 50L)
myAdaboost2$beta_weights
preds2 = myAdaboost2$predict(Ionosphere)
preds2$mmce
#[1] 0.9458689
(nrow(Ionosphere))
#[1] 351

X_Ionosphere = Ionosphere %>% select(-Class)
X_Ionosphere = as.matrix(cbind(intercept = 1, X_Ionosphere))
y_Ionosphere = as.numeric(unlist(as.vector(Ionosphere %>% select(Class))))
ada_compared2 = ada(x = X_Ionosphere, y = y_Ionosphere, iter = 50L)
print(ada_compared2)
#Train Error: 0.017, better than own version

data(Sonar, package = "mlbench")
myAdaboost3 = Adaboost$new(data = Sonar, target = "Class", add_intercept = TRUE)
myAdaboost3$train(baselearner = "treestump", max_iter = 50L)
myAdaboost3$beta_weights
preds3 = myAdaboost3$predict(Sonar)
preds3$mmce
#[1] 0.9855769
(nrow(Sonar))
#[1] 208

X_Sonar = Sonar %>% select(-Class)
X_Sonar = as.matrix(cbind(intercept = 1, X_Sonar))
y_Sonar = as.numeric(unlist(as.vector(Sonar %>% select(Class))))
ada_compared3 = ada(x = X_Sonar, y = y_Sonar, iter = 50L)
print(ada_compared3)
#Train Error: 0.005 , better than own version


data(PimaIndiansDiabetes, package = "mlbench")
myAdaboost4 = Adaboost$new(data = PimaIndiansDiabetes, target = "diabetes", add_intercept = TRUE)
myAdaboost4$train(baselearner = "treestump", max_iter = 50L)
myAdaboost4$beta_weights
preds4 = myAdaboost4$predict(PimaIndiansDiabetes)
preds4$mmce
#[1] 0.7877604
(nrow(PimaIndiansDiabetes))
#[1] 768

X_PimaIndiansDiabetes = PimaIndiansDiabetes %>% select(-diabetes)
X_PimaIndiansDiabetes = as.matrix(cbind(intercept = 1, X_PimaIndiansDiabetes))
y_PimaIndiansDiabetes = as.numeric(unlist(as.vector(PimaIndiansDiabetes %>% select(diabetes))))
ada_compared4 = ada(x = X_PimaIndiansDiabetes, y = y_PimaIndiansDiabetes, iter = 50L)
print(ada_compared4)
#Train Error: 0.098 
## performs pretty much better than own version
