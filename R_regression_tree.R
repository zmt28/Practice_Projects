# import dataset
movie <- read.csv("Desktop/R Files/Files/Movie_regression.csv")
View(movie)

# Data Preprocessing
summary(movie)
movie$Time_taken[is.na(movie$Time_taken)] <- mean(movie$Time_taken,na.rm = TRUE)

# Test-Train Split
install.packages("caTools")
library(caTools)
set.seed(0)
split = sample.split(movie,SplitRatio = 0.8)
train = subset(movie,split==TRUE)
test = subset(movie,split==FALSE)

# Install required packages
install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

# Run regression tree model on train set
regtree <- rpart(formula = Collection~., data = train, control = rpart.control(maxdepth = 3))

# Plot the decision tree
rpart.plot(regtree, box.palette="RdBu", digits = -3)

# Predict Value at any point
test$pred <- predict(regtree, test, type = "vector")

MSE2 <- mean((test$pred - test$Collection)^2)

# Tree Pruning
fulltree <- rpart(formula = Collection~., data = train, control = rpart.control(cp = 0))
rpart.plot(fulltree, box.palette = "RdBu", digits = -3)
printcp(fulltree)
plotcp(regtree)

mincp <- regtree$cptable[which.min(regtree$cptable[,"xerror"]), "CP"]

prunedtree <- prune(fulltree, cp = mincp)
rpart.plot(prunedtree, box.palette = "RdBu", digits = -3)

test$fulltree <- predict(fulltree, test, type = "vector")
MSE2full <- mean((test$fulltree - test$Collection)^2)

test$pruned <- predict(prunedtree, test, type = "vector")
MSE2pruned <- mean((test$pruned - test$Collection)^2)

# Bagging
install.packages('randomForest')
library(randomForest)
set.seed(0)

bagging =randomForest(formula = Collection~., data = train, mtry=17)
test$bagging <- predict(bagging, test)
MSE2bagging <- mean((test$bagging - test$Collection)^2)

# Random Forest
 randomfor <- randomForest(Collection~., data = train,ntree=500)
 
 # Predict Output
 test$random <- predict(randomfor, test)
 MSE2random <- mean((test$random - test$Collection) ^2)
