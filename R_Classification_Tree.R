# Import Dataset
df <- read.csv("Desktop/R Files/Files/Movie_classification.csv")
movie <- read.csv("Desktop/R Files/Files/Movie_regression.csv")
View(df)
install.packages('caTools')
library(caTools)

# Data Preprocessing
summary(df)
df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken,na.rm = TRUE)
# testc <- testc[ -c(20) ]

split = sample.split(movie,SplitRatio = 0.8)
trainc = subset(df,split == TRUE)
testc = subset(df,split == FALSE)


# Run Classification Tree Model on Train Set
classtree <- rpart(formula = Start_Tech_Oscar~., data = trainc, method = 'class', control = rpart.control(maxdepth = 3))

# Plot the Decision Tree
rpart.plot(classtree, box.palette = 'RdBu', digits = -3)

# Predict Value At Any Point
testc$pred <- predict(classtree, testc, type = 'class')

table(testc$Start_Tech_Oscar,testc$pred)

# Bagging
install.packages('randomForest')
library(randomForest)
set.seed(0)
trainc$Start_Tech_Oscar <- as.factor(trainc$Start_Tech_Oscar)
baggingc = randomForest(formula = Start_Tech_Oscar~., data = trainc, method = 'class',ntree=500,mtry=18)
testc$predc <- predict(baggingc, testc, type = "class")

table(testc$Start_Tech_Oscar,testc$predc)

# Random Forest
randomforc <- randomForest(formula = Start_Tech_Oscar~., data = trainc, method = 'class',ntree=500)

# Predict Output
testc$randomc <- predict(randomforc, testc)
table(testc$Start_Tech_Oscar,testc$randomc)
