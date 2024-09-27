rm(list=ls())
load("titanic_train.Rdata")

# PREPROCESSING
# Remove variables that are not useful
titanic.train$Ticket = NULL
titanic.train$Cabin = NULL

# Remove NA 
# In the part 1 of this project, we saw that the were not NA values, however,
# we observed that there were some values in the variable Fare that were 0.
# We assume that they are missing values, so we are going to replace them
# by the mean.

titanic.train$Fare[which(titanic.train$Fare == 0)] = mean(titanic.train$Fare, na.rm = TRUE)

# Let's see now if we need to encode (cast) any of the variables.
str(titanic.train)
# We don't have any categorical data, so everything is fine.

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# TASK 1
# As a complement for the Exploratory Data Analysis we did in the first 
# part of the project, we will use a decision tree to contrast the  the 
# conclusions we draw.

# Create the classification tree
if (!require("rpart")){
  install.packages("rpart")
}
library("rpart")

mytree=rpart(formula=Survived~., data=titanic.train, method="class")

# Plot the classification tree
if (!require("rpart.plot")){
  install.packages("rpart.plot")
}
library("rpart.plot")
prp(mytree,
    type=2,
    extra=106,
    nn=TRUE,
    shadow.col="blue",
    digits=2,
    roundint=FALSE)

# To see the importance of each variable, we can use:
mytree$variable.importance

# Now, before we move on to the second part of this assignment, let's 
# modified the data set. We will replaced the variables Sibsp and Parch
# by another one called "travels_alone" and see if this new variable
# is relevant in the classification tree.
aux = titanic.train$Parch == 0 & titanic.train$SibSp == 0
sum(aux)
travels_alone = rep("No",length(aux))
travels_alone[aux] = "Yes"
titanic.train_2 = cbind(titanic.train, travels_alone)

titanic.train_2$travels_alone = factor(titanic.train_2$travels_alone,
                                     levels = c('No', 'Yes'),
                                     labels = c(0, 1))
titanic.train_2$Parch = NULL
titanic.train_2$SibSp = NULL

mytree2=rpart(formula=Survived~., data=titanic.train_2, method="class")
prp(mytree2, type=2, extra=106, nn=TRUE, shadow.col="blue",digits=2,
    roundint=FALSE)

mytree2$variable.importance
# As we can see (and as we predict in the first assignment), knowing if a
# a passenger traveled or not does not give us much information to determine
# whether he/she survived.

  # Now, we're going to do one last tree with the best combination of 
  # hyperparameters (in the next part we show how to fidn the best combination)
  # of hyperparameters.
my_best_tree =rpart(formula=Survived~., data=titanic.train, method="class",
      control = rpart.control(minsplit = 2, cp = 0, maxdepth = 2.85 ))
prp(my_best_tree, type=2, extra=106, nn=TRUE, shadow.col="blue",digits=2,
    roundint=FALSE)
my_best_tree$variable.importance
# We can see that the results are similar to the ones we obtain previously.


# ---------------------------------------------------------------------
# TASK 2
# ---------------------------------------------------------------------
# First, let's find the best combination of hyperparameters to find the  
# decision tree that fits best with our data.

# Load the libraries
library("rpart")
library("rpart.plot")
library("randomForest")
library("caret")
library("ggplot2")
install.packages("hrbrthemes")   # This one is for a grpah will do later
library(hrbrthemes)

# We are going to use random subsampling
minsplits = seq(2,100,5)
cp =  seq(0, 0.5, 0.1)
maxdepth = seq(1,5)

# We create a data fram with combinations of the hyperparameters
grid = expand.grid(minsplits = minsplits, cp = cp, maxdepth = maxdepth)

# Here we will store the error estimates (accuracy, precision, specificity)
finalmat = matrix(0, 3, nrow(grid))

folds = createFolds(titanic.train$Survived, k = 10)


allmeans = c()
for( k in 1:nrow(grid)){
  insidemat = matrix(0, 10, 3)
  for (i in 1:10){
    training = titanic.train[-folds[[i]],]
    test = titanic.train[folds[[i]],]
    tree =rpart( formula=Survived~., data=training, method="class",
                 control = rpart.control(minsplit = grid$minsplits[k], cp = grid$cp[k], maxdepth = grid$maxdepth[k] ))
    predict = predict(tree, test, type = "class")
    conf_matrix = table(test$Survived ,predict,dnn=c("Actual value","Classifier prediction"))
    accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
    precision = conf_matrix[1,1]/sum(conf_matrix[,1])
    specificity =  conf_matrix[2,2]/sum(conf_matrix[,2])
    insidemat[i,1] = accuracy
    insidemat[i,2] = precision
    insidemat[i,3] = specificity
    
    
  }
  means = colMeans(insidemat)
  #allmeans = c(allmeans,means)
  finalmat[1,k] = means[1] 
  finalmat[2,k] = means[2] 
  finalmat[3,k] = means[3] 
}


grid$acc = c(finalmat[1,])
grid[which.max(finalmat[1,]),]

# Different plots to analyze the results 
plot(finalmat[1,])
ggplot(grid) + aes(x = cp, y = acc, color = as.factor(minsplits)) + geom_line(size = 2) +
  labs(color = "minsplits") + theme(text = element_text(size=18))
ggplot(grid, aes(x=cp, y=acc, color= maxdepth)) + 
  geom_point(size=6) + theme_ipsum()

# We repeated this process multiple times in order to 'reduce' the range
# in which we look for the best hyperparameters, until we found the 
# best combination of the three (it is explained in the report)

# ----------------------------------------------------------------------
# RANDOM FOREST
set.seed(123)
d_mtry = seq(2, 3, 1)
d_ntree = seq(300, 1000, 5)
parameters2 = expand.grid(mtry = d_mtry, ntree = d_ntree)



finalmat3 = matrix(0, 3, nrow(parameters2))


# Numero de arboles -> ntree 
# Numbero de variables que tiene en cuenta en cada arbol -> mtry 
for (y in 1:nrow(parameters2)){
  insidemat = matrix(0, 10, 3)
  for (i in 1:10){
    idx = sample(1:nrow(titanic.train), floor(nrow(titanic.train)*0.8))
    training = titanic.train[idx,] 
    test = titanic.train[-idx,]
    classifier3 = randomForest(formula = Survived~.,
                               data = training,
                               ntree = parameters2[y,2],
                               mtry = parameters2[y,1]
    )
    pred = predict(classifier3,test, type = "class")
    
    conf_matrix = table(test$Survived, pred)
    
    accuracy = sum(diag((conf_matrix))/sum(conf_matrix))
    precision = conf_matrix[1,1]/sum(conf_matrix[,1])
    specificity =  conf_matrix[2,2]/sum(conf_matrix[,2])
    insidemat[i,1] = accuracy
    insidemat[i,2] = precision
    insidemat[i,3] = specificity
  }
  means = colMeans(insidemat)
  #allmeans = c(allmeans,means)
  finalmat3[1,y] = means[1] 
  finalmat3[2,y] = means[2] 
  finalmat3[3,y] = means[3] 
  
}
length(results2)
max2 = which.max(finalmat3[1,] )
parameters2[max2,]



parameters2$acc = c(finalmat3[1,])

ggplot(parameters2) + aes(x = ntree, y = acc, color = as.factor(mtry)) + geom_line(size = 1) +
  labs(color = "mtry") + theme(text = element_text(size=18))


auxvect = c()
for (i in d_ntree){
  auxvect = c(auxvect, mean(parameters2$acc[parameters2$ntree == i]))
  
}
meansofaccuracyntree = data.frame(ntree = d_ntree, acc = auxvect)
ggplot(meansofaccuracyntree) + aes(x = ntree, y = acc) + geom_line(size = 1, color = "green") +
  theme(text = element_text(size=15))

auxvect2 = c()
for (i in d_mtry){
  auxvect2 = c(auxvect2, mean(parameters2$acc[parameters2$mtry == i]))
  
}
meansofaccuracymtry = data.frame(mtry = d_mtry, acc = auxvect2)
ggplot(meansofaccuracymtry) + aes(x = mtry, y = acc) + geom_line(size =1, color = "orange") +
  theme(text = element_text(size=15))

ggplot(parameters2) + aes(y = as.factor(mtry), x = acc) + geom_point(size =1, color = "orange") +
  theme(text = element_text(size=15))







bestclassifier = randomForest(formula = Survived~., data = titanic.train, mtry = 2, ntree = 855)

plot(classifier)
print(classifier)
mymodel = function(test_set){
  
  test_set$Ticket = NULL
  test_set$Cabin = NULL
  
  
  aux2 = which(test_set$Fare == 0)
  meanfare = mean(test_set$Fare[-aux2])
  test_set$Fare[aux2] = rep(meanfare)
  
  test_set$Survived = titanic.train$Survived == 1
  test_set$Survived = as.factor(titanic.train$Survived)
  
  
  pred = predict(bestclassifier, test_set, type = "class")
  conf_matrix = table(test_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  precision = conf_matrix[1,1]/sum(conf_matrix[,1])
  specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
  return(list(prediction = pred, conf_matrix = conf_matrix,accuracy = accuracy,specificity = specificity, precision = precision))
}

save(bestclassifier, mymodel, file = "BestModel.RData")


