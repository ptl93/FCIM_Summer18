##Author: Tuan Le
##Email: tuanle@hotmail.de

library(R6) 
library(MASS)
library(mvtnorm)
## Task:
## Implement quadratic discrimant analysis as explained in lecture

qda = R6Class("quadraticDiscriminantAnalysis",
  public = list(
    data = NULL,
    x.dat = NULL,
    target = NULL,
    classes = NULL,
    n = NULL,
    p = NULL,
    n.j = NULL,
    pi.j = NULL,
    mu.j = NULL,
    sigma.j = NULL,
    predicted.class = NULL,
    predicted.probs = NULL,
    ### CONSTRUCTOR ###
    initialize = function(data, target) {
      self$data = data
      self$target = target
      self$n = nrow(data)
      self$p = ncol(data) - 1
      self$x.dat = as.matrix(self$data[, -which(colnames(data) == self$target)])
      self$classes = levels(data[[target]])
      self$n.j = sapply(self$classes, function(t) sum(data[[self$target]] == t))
    },
    ### METHODS ###
    estimateParams = function() {
      #get class proportion
      self$pi.j = self$n.j / self$n
      #get estimate for mean of features wrt. class
      self$mu.j = lapply(seq.int(length(self$classes)), function(t) {
        idx = (self$data[[self$target]] == self$classes[t])
        res = colSums(x = self$x.dat[idx, ]) / self$n.j[t]
        return(res)
      })
      names(self$mu.j) = self$classes
      #get covariance of features wrt. class
      self$sigma.j = lapply(seq.int(length(self$classes)), function(t) {
        idx = (self$data[[self$target]] == self$classes[t])
        res = matrix(0, nrow = self$p, ncol = self$p)
        for (i in which(idx)) {
          res = res + (self$x.dat[i, ] - self$mu.j[[t]]) %*% t(self$x.dat[i, ] - self$mu.j[[t]])
        }
        res = res / (self$n.j[t] - 1)
        #alternatively: 
        #res = cov(self$x.dat[idx, ])
        return(res)
      })
      names(self$sigma.j) = self$classes
      #estimateParams returns all calculated values
      return(list(pi.j = self$pi.j, mu.j = self$mu.j, sigma.j = self$sigma.j))
    },
    predictQDA = function(newdata) {
      if (!is.element(self$target, colnames(newdata))) {
        newdata[, ncol(newdata) + 1] = NULL
        colnames(newdata)[ncol(newdata)] = self$target
      } else {
        newdata[[self$target]] = NULL
      }
      probs = matrix(nrow = nrow(newdata), ncol = length(self$classes))
      colnames(probs) = self$classes
      for (cl in self$classes) {
        probs[, cl] = dmvnorm(x = newdata, mean = self$mu.j[[cl]], sigma = self$sigma.j[[cl]]) * self$pi.j[[cl]]
      }
      self$predicted.probs = t(apply(probs, 1, function(y) y / sum(y)))
      y = self$classes[max.col(probs)]
      self$predicted.class = factor(y, levels = self$classes)
      #by default predictQDA returns predicted.class
      return(self$predicted.class)
    }
  )
)

set.seed(1)
tr = sample(1:nrow(iris), 25) # 25 indexes for train
train = iris[tr, ]
test = iris[-tr, ]

### MASS Version: ###
z = qda(Species ~ . , data = train)
predictedClass = predict(z,test)$class

### Own Version: ###
#create new object:
myQDA = qda$new(data = train, target = "Species")
print(myQDA)

#estimate the parameters:
myQDA$estimateParams()
print(myQDA)

#predict 
myQDA$predictQDA(newdata = test)
print(myQDA)

##compare
resClass = data.frame(massQDA = predictedClass, myQDA = myQDA$predicted.class)
resClass
sum(resClass$massQDA != resClass$myQDA)

### e.g compare estimated means
z$means
myQDA$mu.j

###### Model Accuracy ###### 

##trainError:
#MASS:
trainPred1 = predict(z)$class
print(paste0("MASS Accuracy for training data: ", sum(trainPred1 == train$Species) / nrow(train)))
#"MASS Accuracy for training data: 1"
#myVersion:
trainPred2 = myQDA$predictQDA(newdata = train)
print(paste0("Accuracy for own version on training data: ", sum(trainPred2 == train$Species) / nrow(train)))
#"Accuracy for own version on training data: 1"

##testError:
#MASS:
print(paste0("MASS Accuracy for test data: ", sum(resClass$massQDA == test$Species) / nrow(test)))
#"MASS Accuracy for test data: 0.856"
#myVersion:
print(paste0("Accuracy for own version on test data: ", sum(resClass$myQDA == test$Species) / nrow(test)))
#[1] "Accuracy for own version on test data: 0.856"
