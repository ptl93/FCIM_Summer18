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
  #get 5 buckets for 1:n
  sampled_idx = sample(1:nrow(X), replace = FALSE, size = nrow(X))
  byf = rep(1:k, each = k)
  #take care if n is not integer-dividable with k 
  ## Helper
  is.wholenumber = function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol
  if (!is.wholenumber(nrow(X)/k)) {
    byf = byf[-length(byf)]
  }
  chunks = split(sampled_idx, f = byf)

  outlist = lapply(1:k, function(i) {
    test_indice = chunks[[i]]
    train_indice = sapply(X = setdiff(1:k, i), function(c) {
      chunks[[c]]
    }, simplify = TRUE)
    train_indice = unlist(train_indice)
    cv_iter = list(train = list(X_train = X[train_indice, ], y_train = y[train_indice]),
      test = list(X_test = X[test_indice,], y_test = y[test_indice]))
    return(cv_iter)
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
#' @param train_size [numeric(1)] train_size fraction. Default is 0.8. Leading to ceiling(train_size*0.8)
#' @return [list(k)] nested list containing each Subsampling iteration Train and Test Data : list(X_train, y_train,) list(X_test, y_test)
#' 
subsampling = function(X, y , B = 50, train_size = 0.8) {
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
iris_CV4 = k_fold_cv(X,y, k = 4)
iris_CV5 = k_fold_cv(X,y, k = 5)
iris_boostrap = bootstrap_sampling(X,y)
iris_subsampling = subsampling(X,y)

## END
