install.packages("devtools")
install.packages("mlr", dependencies = TRUE)
install.packages(c("farff", "OpenML"))

set.seed(78)
eye_data = OpenML::getOMLDataSet(data.id = 1471)
eye_dataset = eye_data$data

## First insigts:
summary(eye_dataset)
## NAs ?
apply(eye_dataset, 2, function(x) sum(is.na(x)))

(n = nrow(eye_dataset))
#[1] 14980
(p = ncol(eye_dataset) - 1)
#[1] 15

### Machine Learning Pipeline Data
idx = sample(1:n, size = 14000, replace = FALSE)

eye_ML = eye_dataset[idx, ]
eye_testing = eye_dataset[setdiff(1:n, idx), ]
table(eye_testing$Class)
#  1   2 
#549 431 
## still balanced

## Preprocess  for later scaling on test##
colmeans = colMeans(eye_ML[, -ncol(eye_ML)])
colstd = apply(eye_ML[, -ncol(eye_ML)], 2, function(col) {
  sd(col)
})

library(mlr)
eye_task = makeClassifTask(id = "eye", data = eye_ML, target = "Class")
print(eye_task)
## Normalize data for speeding up
eye_task = normalizeFeatures(eye_task)
summary(getTaskData(eye_task))

## Get classification learners
classif_lrns = listLearners("classif")

## First insights comparing ML algorithms with default values
gc()
lrns = list(
  makeLearner("classif.fnn", par.vals = list(k = 30)),
  makeLearner("classif.qda"),
  makeLearner("classif.glmnet"),
  makeLearner("classif.boosting"),
  makeLearner("classif.ksvm"),
  makeLearner("classif.xgboost")
)

lrns_h2o = list(
  makeLearner("classif.h2o.randomForest"),
  makeLearner("classif.h2o.gbm")
)

## Create resampling strategy: 10CV
resampling_CV = makeResampleDesc("CV", iters = 10, stratify = TRUE)

## Deploy parallelization
library(parallelMap)
library(parallel)
detectCores()
#[1] 8
#Starting parallelization in mode=socket with cpus=6 within CV learner loop

## Apply resamling strategy for each learner: [this might take a while...]
## Problem with parallelization for h2o library
parallelStartSocket(6)
#Starting parallelization in mode=socket with cpus=6.

CV_results = lapply(lrns, function(learner) {
  res = resample(learner, eye_task, resampling_CV, measures = list(acc, mmce, tpr, fpr, timetrain))
  return(res)
})

parallelStop()
CV_results_h2o = lapply(lrns_h2o, function(learner) {
  res = resample(learner, eye_task, resampling_CV, measures = list(acc, mmce, tpr, fpr, timetrain))
  return(res)
})

CV_results = c(CV_results, CV_results_h2o)
names(CV_results) = c("knn", "qda", "glmnet", "adabag", "ksvm", "xgboost", "randomForest", "gradientboosting")
aggreg_perf = lapply(CV_results, function(x) x$aggr)
aggreg_perf = data.frame(do.call("rbind", aggreg_perf))

library(dplyr)
aggreg_perf = aggreg_perf %>% arrange(desc(acc.test.mean))
print(aggreg_perf)

##save current image
save.image("eyes_ML_s1.Rdata")

load("eyes_ML_s1.Rdata")
library(mlr)
set.seed(80)
## Feature Importance:
listFilterMethods()
feature_importance = generateFilterValuesData(eye_task, method = c("information.gain", "chi.squared"))
plotFilterValues(feature_importance)

## For "true" ML with param tuning take following learners:
## randomForest, xgboost

print(getParamSet(makeLearner("classif.h2o.randomForest")))
rf_ps = makeParamSet(
  makeDiscreteParam("mtries", values = 7:14),
  makeDiscreteParam("ntrees", values = 40:60),
  makeDiscreteParam("max_depth", values = 15:30)
)


print(getParamSet(makeLearner("classif.xgboost")))
xgboost_ps = makeParamSet(
  makeNumericParam("eta", lower = 0.2, upper = 0.6),
  makeNumericParam("gamma", lower = 0, upper = 0.5),
  makeDiscreteParam("max_depth", values = 4:10),
  makeNumericParam("lambda", lower = 1, upper = 2),
  makeNumericParam("rate_drop", lower = 0, upper = 0.3)
)

### Nested resampling for parameter tuning and model evaluation ###
ctrl = makeTuneControlRandom(maxit = 10L)
inner = makeResampleDesc("CV", iters = 4, stratify = TRUE)
outer = makeResampleDesc("CV", iters = 3, stratify = TRUE)
lrn1_rF = makeTuneWrapper("classif.h2o.randomForest", resampling = inner, par.set = rf_ps, control = ctrl,
                          show.info = TRUE, measures = list(acc, tpr, fpr))
lrn2_xgboost = makeTuneWrapper("classif.xgboost", resampling = inner, par.set = xgboost_ps, control = ctrl,
                               show.info = TRUE, measures = list(acc, tpr, fpr))

h2o::h2o.init()
rf_resample = resample(lrn1_rF, eye_task, resampling = outer, extract = getTuneResult, show.info = TRUE)
h2o::h2o.shutdown()
print(rf_resample)
#Resample Result
#Task: eye
#Learner: classif.h2o.randomForest.tuned
#Aggr perf: mmce.test.mean=0.0815713
#Runtime: 440.703
print(rf_resample$extract)
#[[1]]
#Tune result:
#  Op. pars: mtries=8; ntrees=55; max_depth=30
#acc.test.mean=0.9105339,tpr.test.mean=0.9102969,fpr.test.mean=0.0891794

#[[2]]
#Tune result:
#  Op. pars: mtries=7; ntrees=44; max_depth=26
#acc.test.mean=0.9126746,tpr.test.mean=0.9182541,fpr.test.mean=0.0941646

#[[3]]
#Tune result:
#  Op. pars: mtries=7; ntrees=45; max_depth=23
#acc.test.mean=0.9143985,tpr.test.mean=0.9196347,fpr.test.mean=0.0920131

rf_tuneRes = getNestedTuneResultsX(rf_resample)
print(rf_tuneRes)
#mtries ntrees max_depth
#1      8     55        30
#2      7     44        26
#3      7     45        23

## Select thrird configuration
saveRDS(object = rf_resample, file = "randomForest_tuning.Rds")

library(parallelMap)
parallelStartSocket(4L)
xgboost_resample = resample(lrn2_xgboost, eye_task, resampling = outer, extract = getTuneResult, show.info = TRUE)
parallelStop()
print(xgboost_resample)
#Resample Result
#Task: eye
#Learner: classif.xgboost.tuned
#Aggr perf: mmce.test.mean=0.2055703
#Runtime: 1.63134
print(xgboost_resample$extract)
#[[1]]
#Tune result:
#  Op. pars: eta=0.551; gamma=0.376; max_depth=10; lambda=1.12; rate_drop=0.0299
#acc.test.mean=0.7944079,tpr.test.mean=0.8556116,fpr.test.mean=0.2805616

#[[2]]
#Tune result:
#  Op. pars: eta=0.238; gamma=0.186; max_depth=10; lambda=1.06; rate_drop=0.0246
#acc.test.mean=0.7878285,tpr.test.mean=0.8291153,fpr.test.mean=0.2627521

#[[3]]
#Tune result:
#  Op. pars: eta=0.204; gamma=0.113; max_depth=9; lambda=1; rate_drop=0.0765
#acc.test.mean=0.7894802,tpr.test.mean=0.8379052,fpr.test.mean=0.2698340

xgboost_tuneRes = getNestedTuneResultsX(xgboost_resample)
print(xgboost_tuneRes)
#eta     gamma max_depth   lambda  rate_drop
#1 0.5508227 0.3761201        10 1.123866 0.02987101
#2 0.2376101 0.1864396        10 1.064124 0.02457387
#3 0.2044317 0.1131967         9 1.001884 0.07651047
saveRDS(object = xgboost_resample, file = "xgBoost_tuning.Rds")

############# Final Model: take random Forest ############# 
randomForest_learner = setHyperPars(learner = makeLearner("classif.h2o.randomForest"), par.vals = rf_resample$extract[[2]]$x)
h2o::h2o.init()
randomForest_model = train(randomForest_learner, task = eye_task)

## Predict on test set:: ##
## First scale test data with colMeans and colSd
for (i in 1:length(colmeans)) {
  eye_testing[, i] = (eye_testing[, i] - colmeans[i]) / colstd[i]
}
summary(eye_testing)

test_pred = predict(obj = randomForest_model, newdata = eye_testing)
calculateConfusionMatrix(test_pred)
acc = sum(test_pred$data$truth == test_pred$data$response)/nrow(eye_testing)
print(paste("Accuracy on test data:", round(acc, 4)))
#[1] "Accuracy on test data: 0.9296"

## Save final Random Forest
saveRDS(object = randomForest_model, "randomForest_finalModel.Rds")

## END ##      