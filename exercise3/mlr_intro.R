### Exercise 3 - Task 2 - mlr Intro / Tutorial

##Tuan Le,
##tuanle@hotmail.de

library(mlr)

#2b)
(all_performance_measures = listMeasures())
#filter according to learning problem
(all_performance_measures_classif = listMeasures(obj = "classif"))
(all_performance_measures_regression = listMeasures(obj = "regr"))
(all_performance_measures_cluster = listMeasures(obj = "cluster"))

#2c) Use the bh.task regression task from mlr and split the data into 50% training data and 50% test data while
#training and predicting (i.e., use the subset argument of the train and predict function). Fit a prediction
#model (e.g. CART) to the training set and make predictions for the test set.

## Get Task
bh.task = bh.task
## Get Task Data to get first insights
bh.data = getTaskData(bh.task)
n = getTaskSize(bh.task)
head(bh.data)
summary(bh.data)

## Get all listed regression learners
##https://mlr-org.github.io/mlr-tutorial/release/html/integrated_learners/#regression-61 to check which learners are available
regr_learners = listLearners("regr")

## Initiliaze the learners
lrn1 = makeLearner("regr.rpart")
lrn2 = makeLearner("regr.randomForest")
lrn3 = makeLearner("regr.glmnet")

## Train and Test Set splitting
## Use 1/2 of the observations for training
set.seed(1)
train.set = sample(n, size = n/2)
test.set = setdiff(1:n, train.set)

## Train the learner
model1 = train(lrn1, bh.task, subset = train.set)
model2 = train(lrn2, bh.task, subset = train.set)
model3 = train(lrn3, bh.task, subset = train.set)

## Predict on test set
pred1 = predict(model1, task = bh.task, subset = test.set)
pred2 = predict(model2, task = bh.task, subset = test.set)
pred3 = predict(model3, task = bh.task, subset = test.set)

## Get Performance
performance1 = performance(pred1, measures = list(mse, mae, medse, rmse))
performance2 = performance(pred2, measures = list(mse, mae, medse, rmse))
performance3 = performance(pred3, measures = list(mse, mae, medse, rmse))

all_performances = rbind(performance1, performance2, performance3)
row.names(all_performances) = c("CART", "RandomForest", "GLMNet")
print(all_performances)

#2d) Now use different observations (but still 50% of them) for the training set. What effects does this have?
## Apply resampling method Subsampling with 5 iterations each 50 % train 50% test
rdesc = makeResampleDesc("Subsample", iters = 5, split = 1/2)

## Calculate the performance measures for each learners
all_lrns = list(cart = lrn1, rf = lrn2, glmnet = lrn3)

resampling_results = lapply(all_lrns, function(lrn) {
  resample(lrn, bh.task, rdesc, measures = list(mse, mae, medse, rmse, timetrain))
})

#Cart still performs best in aggregated results


#(2e) What do you have to do in mlr if you want the prediction output to be probabilities (and not classes)?
## Generate the task
iris.task = iris.task

## Generate the learner
lrn1_class = makeLearner("classif.lda", predict.type = "response")
lrn1_prob = makeLearner("classif.lda", predict.type = "prob")

## Train the learner on whole dataset
mod1_class = train(lrn1_class, iris.task)
mod1_prob = train(lrn1_prob, iris.task)

pred1_class = predict(mod1_class, iris.task)
pred1_prob = predict(mod1_prob, iris.task)

all_pred = cbind(as.data.frame(pred1_class), (as.data.frame(pred1_prob)))
head(all_pred)
