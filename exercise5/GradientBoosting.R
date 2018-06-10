## Tutorial 5 ##
## Tuan Le ##

## Exercise 1: Implementation of simple gradient boosting algorithm
## Settings: 
# - can handle Regression task
# - use L2-loss
# - use decision stumps (trees of depth one) as base learner
# - use linesearch to find the optimal "step size"

library(R6)
library(rpart)

GradientBoosting = R6Class("GradientBoosting",
  public = list(
    X = NULL,
    y = NULL,
    n = NULL,
    p = NULL,
    data = NULL,
    target = NULL,
    beta_weights = NULL,
    base_models = NULL,
    f_hat = NULL,
    trace = NULL,
    linesearch_solutions = NULL,
    formula = NULL,
    f_0 = NULL,
    ### CONSTRUCTOR ###
    #' Initializes Adaboost object
    #' @param data data.frame
    #' @param target
    #' @param add_intercept logical, whether intercept should be added. Default is FALSE
    #' @param formula formula for baselearner. Default is null. If null is inserted full model target ~ . will be taken
    #' @return GradientBoost object
    initialize = function(data, target, add_intercept = FALSE, formula = NULL) {
      if (add_intercept) {
        data = cbind(intercept = 1, data)
        self$add_intercept = TRUE
      } 
      self$X = data[, -which(names(data) == target)]
      self$y = data[[target]]
      if (!is.numeric(self$y)) stop("Target variable is not numeric. Current version only supports regression task.")
      self$data = data
      self$n = nrow(data)
      self$p = ncol(data) - 1
      if (is.null(formula)) {
        self$formula = as.formula(paste(target, " ~ ."))
      } else {
        self$formula = formula
      }
    },
    ### METHODS ###
    #' Trains GradientBoosting object
    #' @param baselearner weak baselearner used for GradientBoosting Default is a tree strump with max.depth=1 from rpart
    #' @param M maximal iteration for GradientBoosting
    #' @param ... additional parameters passed to baselearner algorithm
    #' @return NULL
    train = function(baselearner = "treestump", M = 30L, ...) {
      #init base models list to store
      self$base_models = vector("list", length = M + 1)
      #init f_iter predictions list to store. M+1 to include iteration 0
      self$f_hat = vector("list", length = M + 1)
      #init beta weights
      self$beta_weights = numeric(M + 1)
      #init trace list for matrices
      self$trace = vector("list", length = M + 1)
      #init line search solutions
      self$linesearch_solutions = vector("list", length = M + 1)
      if (baselearner == "treestump") {
        for (m in 1:M) {
          # Get prediction value for additive model 
          self$f_hat[[m]] = self$additive_model_pred(m - 1)
          # Compute residuals for every observation. Residuals are negative gradient of loss from additive model
          pseudo_resids = self$L2_deriv(self$y, self$f_hat[[m]])
          # Create residum data.frame to pass into rpart
          resid_data = data.frame(X = self$X, pseudo_resids = pseudo_resids)
          # Fit base model for pseuo-rediduals
          self$base_models[[m]] = rpart(pseudo_resids ~ ., data = resid_data, control = rpart.control(maxdepth = 1))
          # Predict current base learner
          base_pred_hat = predict(obj = self$base_models[[m]], newdata = resid_data)
          # Get optimal beta for m-th iteration
          self$beta_weights[m] = self$line_search(y_hat = self$f_hat[[m]], base_pred_hat = base_pred_hat, j = m)
          # Trace 
          #trace[[m]] = data.frame(X = self$X, y = self$y, y_hat = self$f_hat[[m]],
          #  pseudo_resids = pseudo_resids, base_pred_hat = base_pred_hat)
        }
      }
    return(invisible(NULL))
    },
    #' Computes the prediction of the additive model f_hat_{m-1}
    #' @param m current iteration 0,1,..,M
    #' @return prediction of the additive model f_hat_{m-1} including all base learner models unti m-1 on data X 
    additive_model_pred = function(m) {
      ## Init: For the first gradient boosting prediction as predicted value use the mean of target variable
      if (m == 0) {
        return(rep(mean(self$y), self$n))
      } else {# If not initialization take all (m-1) base learners in consideration for f_hat:
        # Take (m-1) predictions for each base learner, multiply it with the base learner weights beta and then sum up
         preds_base_models = sapply(1:m, function(b) {
          predict(obj = self$base_models[[b]], newdata = self$data)
        })
         preds_base_models = preds_base_models %*% self$beta_weights[1:m] + self$f_hat[[1]]
      }
    },
    #' Computes the negative derivative of L2-loss function: vectorized version
    #' @param y true target vector 
    #' @param f_xi predicted target vector
    #' @return negative gradient of loss function L(x, f(x)) forall observations x_i, i = 1,...n
    L2_deriv = function(y, f_hat) {
      2*(y - f_hat)
    },
    #' Line search: Computes the optimal m-th beta for gradient descent solving univariate problem
    #' @param y_hat true target vector 
    #' @param f_xi predicted target vector
    #' @param j current gradient boostigng algorithm (m-1)
    #' @return optimal beta weight for base learner m.
    line_search = function(y_hat, base_pred_hat, j) {
      # L2 loss : (y, f_m) = sum_i (y_i - (f_{m-1} + beta * f_m))^2
      obj = function(beta) {
        L2_inner_sum = t(self$y - (y_hat + self$beta_weights * base_pred_hat)) %*% (self$y - (y_hat + self$beta_weights * base_pred_hat))
        return(L2_inner_sum)
      }
      # find best beta
      self$linesearch_solutions[[j]] = optimize(obj, interval = c(0, 10000))
      return(self$linesearch_solutions[[j]]$minimum)
    }
  )
)



set.seed(9)
X = seq(from = 0, to = 6, by = 0.1)
y = sin(x) + rnorm(n = length(X), mean = 0, sd = 0.10)
data = data.frame(X = X, y = y)
#plot(x, y, type = "p", xlab = "x", ylab = "sin(x) + N(0, 0.10)")

myGradientBoosting = GradientBoosting$new(data = data, target = "y")
myGradientBoosting
myGradientBoosting$train(baselearner = "treestump", M = 50L)

