### Kernel Support Vector Machine Application for the Ionosphere dataset ###
### Tuan Le
### tuanle@hotmail.de

# Load data and get specific characteristics
data("Ionosphere", package = "mlbench")
summary(Ionosphere)
n = nrow(Ionosphere)
p = ncol(Ionosphere) - 1

print(levels(Ionosphere$Class))
sum(is.na(Ionosphere$Class))
#0

##V2 is a factor wih only 1 level. Drop that since it does not give any hint about the underlying function since its constant
Ionosphere = Ionosphere[, -2]
p = ncol(Ionosphere) - 1
## Encode -1 and 1 for SVM
Ionosphere$Class = ifelse(Ionosphere$Class == "good", 1, -1)
Ionosphere$Class = as.factor(Ionosphere$Class)
levels = levels(Ionosphere$Class)
nlevels = length(levels)

library(mlr)
iono_task = makeClassifTask("Ionosphere", data = Ionosphere, target = "Class")
iono_task = normalizeFeatures(iono_task)

## Apply nested resampling for clean evaluation:
## Outer loop: 6 fold CV
## Inner loop: 5 fold CV
## Each cross validation iteration applies a random search with 5 iterations
## Total runs: 6*5*5 = 150 runs
outer_CV = makeResampleDesc("CV", iters = 6L)
inner_CV = makeResampleDesc("CV", iters = 5L)
random_search = makeTuneControlRandom(maxit = 5L)

## Define parameter set for random search. Only apply radial basis kernel
ksvm_parameters = (print(getParamSet("classif.ksvm")))

ksvm_paramset = makeParamSet(
  makeDiscreteParam("C", values = 2^(-4:4)),
  makeDiscreteParam("sigma", values = 2^(-6:6))
)

## Define tuned learner with inner resampling strategy
ksvm_lrn = makeTuneWrapper(learner = makeLearner("classif.ksvm"), resampling = inner_CV, 
                           control = random_search, par.set = ksvm_paramset, 
                           measures = list(acc, mmce, fpr, tpr), show.info = TRUE)

## load parallelization libraries
library(parallel)
library(parallelMap)
detectCores()
#Linux
parallelStartMulticore(cpus = 2L)

set.seed(920)
ksvm_resample = resample(ksvm_lrn, iono_task, resampling = outer_CV, extract = getTuneResult, show.info = TRUE)
#Aggregated Result: mmce.test.mean=0.0825541
parallelStop()

print(ksvm_resample)
print(ksvm_resample$extract)
#[[1]]
#Tune result:
#  Op. pars: C=4; sigma=0.015625
#acc.test.mean=0.9421391,mmce.test.mean=0.0578609,fpr.test.mean=0.0113238,tpr.test.mean=0.8533734
#[[2]]
#Tune result:
#  Op. pars: C=1; sigma=0.015625
#acc.test.mean=0.9350088,mmce.test.mean=0.0649912,fpr.test.mean=0.0170439,tpr.test.mean=0.8518519
#[[3]]
#Tune result:
#  Op. pars: C=16; sigma=0.125
#acc.test.mean=0.9656341,mmce.test.mean=0.0343659,fpr.test.mean=0.0391640,tpr.test.mean=0.9720824
#[[4]]
#Tune result:
#  Op. pars: C=1; sigma=0.0625
#acc.test.mean=0.9623027,mmce.test.mean=0.0376973,fpr.test.mean=0.0100063,tpr.test.mean=0.9095639
#[[5]]
#Tune result:
#  Op. pars: C=0.25; sigma=0.125
#acc.test.mean=0.9317943,mmce.test.mean=0.0682057,fpr.test.mean=0.0599828,tpr.test.mean=0.9307706
#[[6]]
#Tune result:
#  Op. pars: C=1; sigma=0.5
#acc.test.mean=0.8943308,mmce.test.mean=0.1056692,fpr.test.mean=0.1599704,tpr.test.mean=0.9913043

## Get parameters for each outer CV iteration
ksvm_tuneRes = getNestedTuneResultsX(ksvm_resample)
ksvm_tuneRes = cbind(ksvm_tuneRes, ksvm_resample$measures.test)
print(ksvm_tuneRes)
#   C    sigma iter       mmce
#1  4.00 0.015625    1 0.05172414
#2  1.00 0.015625    2 0.03389831
#3 16.00 0.125000    3 0.13559322
#4  1.00 0.062500    4 0.10169492
#5  0.25 0.125000    5 0.05172414
#6  1.00 0.500000    6 0.12068966

## Extract that setting with minimal mmce
optimal_setting = getNestedTuneResultsX(ksvm_resample)[which.min(ksvm_tuneRes$mmce),]
print(optimal_setting)
#  C    sigma
#2 1 0.015625

##END##
