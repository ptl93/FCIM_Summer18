### Implementation of k-nearest neighbors algorithm
### tuanle@hotmail.de

## Algorithm: https://www.researchgate.net/figure/260397165_Pseudocode-for-KNN-classification

library(R6)

k_nearest_neighbors = R6Class("k_nearest_neighbors",
  public = list(
    X = NULL,
    y = NULL,
    n = NULL,
    p = NULL,
    g = NULL,
    nearest_neighbors = NULL,
    classification = FALSE,
    y_pred = NULL,
    ### Constructor ####
    # @param X [matrix(n,p)] or [data.frame(n,p)]
    #   Training matrix / dataframe for knn algorithm
    # @param y [numeric(n)] or [factor(n)] target vector depending on ml-problem. 
    #   returns k_nearest_neighbors instance
    initialize = function(X, y) {
      self$X = X
      self$y = y
      self$n = nrow(X)
      self$p = ncol(X)
      # if classification problem, get number of classes
      if (is.factor(y)) {
        self$g = length(levels(y))
        self$classification = TRUE
      } 
    },
    ### Train the non-parametric algorithm ###
    # @param k [integer(1)]. Default is to take 5 nearest neighbors
    #   Number of neighbors considered for classification/regression.
    # @return [invisible(NULL)]. 
    #   returns nothing. Results will be saved within k_nearest_neighbors instance.
    # for neighborhood check instance$nearest_neighbors
    # for prediction check instance$y_pred
    train_knn = function(k = 5, X_data = NULL, y_data = NULL){
      if (is.null(X_data) & is.null(y_data)) {
        X_data = self$X
        y_data = self$y
      }
      self$y_pred = rep(NA, times = nrow(X_data))
      self$nearest_neighbors = vector("list", length = nrow(X_data))
      # For each observation in X(_train) compute the distance to each other observation
      for (observation in seq.int(nrow(X_data))) {
        #out: nx1 vector, because n instances in train.
        dist = apply(X = self$X, MARGIN =  1, function(x) sum((x - X_data[observation, ])^2))
        # order by returning index of observations for distance in increasing order. 
        dist = order(dist, decreasing = FALSE)
        # take the k-nearest neighbors
        nearest_neighbors_idx = dist[1:k]
        # store into nearest_neighbors_list
        self$nearest_neighbors[[observation]] = nearest_neighbors_idx
        ### As of now code predict function within train
        if (self$classification) {
          ## If classification problem, make majority vote for all observation within a neighbourhood
          class_frequency_in_neighborhood = table(self$y[nearest_neighbors_idx])
          majority_vote = names(class_frequency_in_neighborhood)[which.max(class_frequency_in_neighborhood)]
          ## store prediction in y_train_pred vector
          self$y_pred[observation] = majority_vote
        } else {
          ## If regression, take the mean of the neighbor hood values
          mean_value_neighbors = mean(self$y[nearest_neighbors_idx])
          self$y_pred[observation] = mean_value_neighbors
        }
      }
      if (self$classification) {
        print(paste("Accuracy:", sum(y_data == self$y_pred)/nrow(X_data)))
      } else {
        print(paste("Accuracy (MSE):", sum((y_data - self$y_pred)^2)/nrow(X_data)))
      }
      return(invisible(NULL))
    },
    ### Predicts outcome for test data ###
    # @param X_test [matrix] or [data.frame] to predict target vector
    # @param y_test [numeric] or [factor] test target vector
    # @param k [integer(1)]  Number of neighbors considered for classification/regression.
    #   returns nothing. Results will be saved within k_nearest_neighbors instance.
    # for neighborhood check instance$nearest_neighbors
    # for prediction check instance$y_pred
    predict_knn = function(k = 5, X_test, y_test) {
      #call train_knn
      self$train_knn(k = k, X_data = X_test, y_data = y_test)
    }
  )
)

X = iris[, 1:4]
y = iris$Species

### Hold-Out-Sampling:
set.seed(1)
train_idx = sample(1:nrow(X), size = ceiling(2/3*nrow(X)), replace = FALSE)
test_idx = setdiff(1:nrow(X), train_idx)
X_train = X[train_idx,]
y_train = y[train_idx]
X_test = X[test_idx,]
y_test = y[test_idx]

knn_iris = k_nearest_neighbors$new(X = X_train, y = y_train)
## check init values
print(knn_iris$classification)
## train
knn_iris$train_knn(k = 5)
#[1] "Accuracy: 0.96"
print(knn_iris$y_pred)
## test
knn_iris$predict_knn(k = 5, X_test = X_test, y_test = y_test)
##[1] "Accuracy: 1"
sum(knn_iris$y_pred == y_test)/nrow(X_test)

## try regression
X = mtcars[,-1]
y = mtcars[,1]
### Hold-Out-Sampling:
set.seed(1)
train_idx = sample(1:nrow(X), size = ceiling(2/3*nrow(X)), replace = FALSE)
test_idx = setdiff(1:nrow(X), train_idx)
X_train = X[train_idx,]
y_train = y[train_idx]
X_test = X[test_idx,]
y_test = y[test_idx]

knn_cars = k_nearest_neighbors$new(X = X_train, y = y_train)
## check init values
print(knn_cars$classification)
## train
knn_cars$train_knn(k = 3)
#[1] "Accuracy (MSE): 5.42909090909091"
print(knn_cars$y_pred)
## test
knn_cars$predict_knn(k = 3, X_test = X_test, y_test = y_test)
#[1] "Accuracy (MSE): 2.24811111111111"
knn_cars$y_pred

## END
