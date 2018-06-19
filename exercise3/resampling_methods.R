### Implementation of Resampling Methods ###
### Tuan Le, tuanle@hotmail.de ###


#' Computes hold-out sampling for a dataset
#' @param X [matrix(n,p)] or [data.frame(n,p)] Feature input data 
#' @param y [numeric(n)] or [factor(n)] Target input data
#' @param train_size [numeric(1)] train_size fraction. Default is 2/3. Leading to ceiling(train_size*2/3)
#' @return [list(2)] nested list containing list(X_train, y_train,) list(X_test, y_test)
#' 
holdout_sampling = function(X, y, train_size = 2/3){
  #Sample train_idx without replacement
  train_idx = sample(x = 1:nrow(X), size = ceiling(2/3*nrow(X)), replace = FALSE)
  #Get test_idx
  test_idx = setdiff(1:nrow(X), train_idx)
  #Create outlist with filtered train and test data
  outlist = list(train = list(X_train = X[train_idx,], y_train = y[train_idx]), test = list(X_test = X[test_idx,], y_test = y[test_idx]))
  return(outlist)
}

#' Computes k-fold crossvalidation 
#' @param X [matrix(n,p)] or [data.frame(n,p)] Feature input data 
#' @param y [numeric(n)] or [factor(n)] Target input data
#' @param k [integer(1)] number of cross validation iterations. Default is k = 5
#' @return [list(k)] nested list containing each cross validation iteration folds : list(X_train, y_train,) list(X_test, y_test)
#' 
k_fold_cv = function(X, y , k = 5) {
  outlist = lapply(1:k, function(f) {
    holdout_sampling(X = X, y = y, train_size = (nrow(X) - nrow(X)/k))
  })
  names(outlist) = paste("Fold", 1:k)
  return(outlist)
}

#' Computes Bootstrap resampling
#' @param X [matrix(n,p)] or [data.frame(n,p)] Feature input data 
#' @param y [numeric(n)] or [factor(n)] Target input data
#' @param B [integer(1)] number of Bootstrap iterations. Default is B = 50
#' @return [list(k)] nested list containing each Bootstrap iteration Train and Test Data : list(X_train, y_train,) list(X_test, y_test)
#' 
bootstrap_sampling = function(X, y , B = 50) {
  outlist = lapply(1:B, function(b) {
    train_idx = sample(1:nrow(X), size = nrow(X), replace = TRUE)
    test_idx = setdiff(1:nrow(X), train_idx)
    bootrstrap_list = list(train = list(X_train = X[train_idx,], y_train = y[train_idx]), test = list(X_test = X[test_idx,], y_test = y[test_idx]))
  })
  names(outlist) = paste("Bootstrap Iteration:", 1:B)
  return(outlist)
}

#' Computes Subsampling resampling
#' @param X [matrix(n,p)] or [data.frame(n,p)] Feature input data 
#' @param y [numeric(n)] or [factor(n)] Target input data
#' @param B [integer(1)] number of Subsampling iterations. Default is B = 50
#' @param train_size [numeric(1)] train_size fraction. Default is 2/3. Leading to ceiling(train_size*2/3)
#' @return [list(k)] nested list containing each Subsampling iteration Train and Test Data : list(X_train, y_train,) list(X_test, y_test)
#' 
subsampling = function(X, y , B = 50, train_size = 2/3) {
  outlist = lapply(1:B, function(b) {
    train_idx = sample(1:nrow(X), size = nrow(X), replace = TRUE)
    test_idx = setdiff(1:nrow(X), train_idx)
    subsampling = holdout_sampling(X,y, train_size)
  })
  names(outlist) = paste("Subsampling Iteration:", 1:B)
  return(outlist)
}


X = iris[,-5]
y = iris[, 5]

iris_holdout = holdout_sampling(X, y)
iris_CV = k_fold_cv(X,y)
iris_boostrap = bootstrap_sampling(X,y)
iris_subsampling = subsampling(X,y)

## END
