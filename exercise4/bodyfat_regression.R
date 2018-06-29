## Tutorial 4 ##
## Tuan Le ##

#Exercise 1:
set.seed(26)
library(mlr)

#load bodyfat data from TH.data package
data("bodyfat", package = "TH.data")
#get first view of data
summary(bodyfat)

p = ncol(bodyfat) - 1
### a) create a mlr regression task
bodyfat_task = makeRegrTask(id = "bodyfat", data = bodyfat, target = "DEXfat")
print(bodyfat_task)
#only numerics in feature data

### b) fit a regression tree with complexity param 0.05 and minimum number of 15 observations for a split
#first check the control hyperparameters from rpart function:
library(rpart)
?rpart.control

#init learner
regr_tree_learner = makeLearner("regr.rpart", par.vals = list(minsplit = 15, cp = 0.05))

#train learner using complete dataset
regr_tree_mod = train(regr_tree_learner, bodyfat_task)

#visualize the predictions for the features waistcirc and anthro3c seperately as well as in one plot
#extract model from wrapped model
reg_tree_model = regr_tree_mod$learner.model
#get variable_importance
variable_importance = reg_tree_model$variable.importance
print(variable_importance)
#waistcirc      hipcirc     anthro3c      anthro4     anthro3a     anthro3b  kneebreadth elbowbreadth 
#6317.0737    5551.1854    4729.2299    4382.0164    4199.4853    4089.7272     274.4103     219.5283

library(rpart.plot)
rpart.plot(reg_tree_model, type = 2, extra = 101)
#one plot
tree_waist_anth = snip.rpart(reg_tree_model, toss = 3)
rpart.plot(tree_waist_anth, type = 2, extra = 101)

#seperately
tree_only_waistcirc = snip.rpart(reg_tree_model, toss = c(2,3))
rpart.plot(tree_only_waistcirc)

tree_only_anthro3c = snip.rpart(reg_tree_model, toss = c(1,3))
rpart.plot(tree_only_anthro3c)


### use mlr plotlearnerpred
plotLearnerPrediction(regr_tree_learner, bodyfat_task, features = "waistcirc")
plotLearnerPrediction(regr_tree_learner, bodyfat_task, features = "anthro3c")
plotLearnerPrediction(regr_tree_learner, bodyfat_task, features = c("waistcirc", "anthro3c"))
## --> for each rectangle we get one prediction. In this case only 5 predictions
### d) 10-fold-cv and calc mean_se and median_ae
rdesc = makeResampleDesc("CV", iters = 10)
print(rdesc)

listMeasures("regr")
str(mse) #"Mean of squared errors"
str(mae) #"Mean of absolute errors"
str(medse) #"Median of squared errors"
str(medae) #"Median of absolute errors"

r = resample(regr_tree_learner, bodyfat_task, rdesc, measures = list(mse, medae))
(resample_measures_test = r$measures.test)
## performance measures can differ pretty much in cross validation iterations e.g iteration 2 and 8
#iter      mse    medae
#1     1 19.61648 3.467500
#2     2 16.02344 2.658000
#3     3 68.79671 6.050000
#4     4 27.84412 2.937826
#5     5 67.71968 6.156471
#6     6 37.02922 4.972174
#7     7 20.12763 3.296364
#8     8 59.99467 4.542667
#9     9 32.77542 5.005000
#10   10 57.33333 8.629091

print(r$aggr)
#mse.test.mean medae.test.mean 
#40.726070        4.771509 

### More advanced using regression tree as well: ###
#normalize features
bodyfat_task_normalized = normalizeFeatures(bodyfat_task)

#parameter tuning:
#have a look at control hyper parameter again
?rpart.control

#define parameters to tune
parameter_set = makeParamSet(
  makeDiscreteParam("minsplit", values = 10:25),
  makeNumericParam("cp", lower = 0.01, upper = 0.1)
)

#define tuning method: random search with 100L iterations
ctrl = makeTuneControlRandom(maxit = 100L)

#perform tuning using 10 cross validation again
tune_res = tuneParams("regr.rpart", task = bodyfat_task_normalized, resampling = rdesc,
  par.set = parameter_set, control = ctrl, measures = list(mse, medae))

#[Tune] Result: minsplit=14; cp=0.0129 : mse.test.mean=27.8243633,medae.test.mean=2.9910237

#regression tree with defined settings is worst in aggregated 10-fold cross validation
print(r$aggr)
#mse.test.mean medae.test.mean 
#40.726070        4.771509
