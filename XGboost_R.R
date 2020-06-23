# Import Data
df <- read.csv("Desktop/R Files/Files/Movie_classification.csv")
movie <- read.csv("Desktop/R Files/Files/Movie_regression.csv")

df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken,na.rm = TRUE)

split = sample.split(movie,SplitRatio = 0.8)
trainc = subset(df,split == TRUE)
testc = subset(df,split == FALSE)
# XGBoost
install.packages('xgboost')
library(xgboost)

trainY = trainc$Start_Tech_Oscar == "1"

trainX <- model.matrix(Start_Tech_Oscar ~ .-1, data =  trainc)
# trainX <- trainX[,-12]
testY = testc$Start_Tech_Oscar == '1'

testX <- model.matrix(Start_Tech_Oscar ~ .-1, data = testc)
# testX <- testX[,-12]
# Delete additional variable

Xmatrix <- xgb.DMatrix(data = trainX, label = trainY)
Xmatrix_t <- xgb.DMatrix(data = testX, label = testY)

Xgboosting <- xgboost(data = Xmatrix, # the data
                      nround = 50, # Max num of boosting iterations
                      objective = "multi:softmax", eta = 0.3, num_class = 2, max_depth = 100)
xgpred <- predict(Xgboosting, Xmatrix_t)
table(testY, xgpred)
