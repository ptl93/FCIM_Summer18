## Tutorial 4 ##
## Tuan Le ##

#Exercise 2:
library(mlbench)
library(mlr)
set.seed(26)
?mlbench::mlbench.spirals
spirals = mlbench.spirals(n = 500, cycles = 1, sd = 0.2)
spirals_data = as.data.frame(spirals)

summary(spirals_data)
library(ggplot2)
class_plot = ggplot(spirals_data, aes(x = x.1, y = x.2, shape = classes, color = classes)) + geom_point()
print(class_plot)

spiral_task = makeClassifTask("spiral", spirals_data, target = "classes")
print(spiral_task)

?plotLearnerPrediction
plotLearnerPrediction("classif.lda", task = spiral_task)
plotLearnerPrediction("classif.qda", task = spiral_task)
plotLearnerPrediction("classif.rpart", task = spiral_task)
plotLearnerPrediction("classif.ada", task = spiral_task)
plotLearnerPrediction("classif.randomForest", task = spiral_task)
#random forest clearly overfitted
