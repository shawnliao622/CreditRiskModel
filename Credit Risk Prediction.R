# Read the dataset
data <- read.csv("credit3.csv")

############## K-Nearest Neighbors ##########

Credit <- data
Credit$PROFITABLE <- ifelse(Credit$NPV > 0, 1, 0)
Credit$AMOUNT_REQUESTED <- as.numeric(gsub(',', '',Credit$AMOUNT_REQUESTED))
Credit$CREDIT_EXTENDED <- NULL
Credit$NPV <- NULL

# create dummy variables for CHK_ACCT, SAV_ACCT, HISTORY, JOB and TYPE
library(fastDummies)
Credit <- dummy_cols(Credit, select_columns = 
                       c('CHK_ACCT', 'SAV_ACCT', 'HISTORY', 'JOB', 'TYPE'))
Credit[, c(1,3:4,7,10,20)] <- NULL

# Normalize each variable
fun <- function(x){ 
  a <- mean(x) 
  b <- sd(x) 
  (x - a)/(b) 
} 
Credit[,1:15] <- apply(Credit[,1:15], 2, fun)
Credit[,17:41] <- apply(Credit[,17:41], 2, fun)

# Split the sample into training (70%) and validation (30%) samples 
# with the seed set at 12345. 
set.seed(12345)
inTrain <- sample(nrow(Credit), 0.7*nrow(Credit))
#
dftrain <- data.frame(Credit[inTrain,])
dfvalidation <- data.frame(Credit[-inTrain,])

library(class)
train_input <- as.matrix(dftrain[,-16])
train_output <- as.vector(dftrain[,16])
validate_input <- as.matrix(dfvalidation[,-16])
#
# Predicting Profitability using K-nearest neighbors
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)
#
set.seed(3)
for (i in 1:kmax){
  prediction <- knn(train_input, train_input, train_output, k=i)
  prediction2 <- knn(train_input, validate_input, train_output, k=i)
  #
  # The confusion matrix for training data is:
  CM1 <- table(prediction, dftrain$PROFITABLE)
  # The training error rate is:
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  # The confusion matrix for validation data is: 
  CM2 <- table(prediction2, dfvalidation$PROFITABLE)
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
}
plot(c(1,kmax),c(0,0.5),type="n", xlab="k",ylab="Error Rate")
lines(ER1,col="red")
lines(ER2,col="blue")
legend(10, 0.47, c("Training","Validation"),lty=c(1,1), col=c("red","blue"))

# Find the value of k where the error rate is minimized
z <- which.min(ER2)
cat("Minimum Validation Error k:", z)

# The error rate at the best k.
cat("Error Rate:", ER2[z])

# Compute the confusion matrix at the best k
predict_validation <- knn(train_input, validate_input, train_output, k=z)
(table1 <- table(dfvalidation$PROFITABLE, predict_validation))

# The sensitivity at the best k.
(sensitivity <- table1[2,2] / sum(table1[2,1:2]))

# The specificity at the best k.
(specificity <- table1[1,1] / sum(table1[1,1:2]))

# Plot the lift chart for the classifier using the validation data at best k
prediction1 <- knn(train_input, validate_input, train_output, k=z, prob=T)
predicted.probability <- attr(prediction1, "prob")
df1 <- data.frame(prediction1, predicted.probability,dfvalidation$PROFITABLE)

df1S <- df1[order(-predicted.probability),]
df1S$Gains <- cumsum(df1S$dfvalidation.PROFITABLE)
plot(df1S$Gains,type="n",main="Lift Chart",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains,col="blue")
abline(0,sum(df1S$dfvalidation.PROFITABLE)/nrow(df1S),lty = 2, col="red")

############## Naive Bayes ##########

# Data pre-processing
Credit_NB <- data
Credit_NB[,1] <- NULL
Credit_NB$PROFITABLE <- ifelse(Credit_NB$NPV > 0, 1, 0)
Credit_NB$AMOUNT_REQUESTED <- as.numeric(gsub(',', '',Credit_NB$AMOUNT_REQUESTED))
Credit_NB$CREDIT_EXTENDED <- NULL
factor_names <- c(2:4,6:19)
Credit_NB[,factor_names] <- lapply(Credit_NB[,factor_names] , factor)
Credit_NB$NPV <- NULL

# Split data into train and validation
set.seed(12345)
inTrain_NB <- sample(nrow(Credit_NB), 0.7*nrow(Credit_NB))
#
dftrain_NB <- data.frame(Credit_NB[inTrain_NB,])
dfvalidation_NB <- data.frame(Credit_NB[-inTrain_NB,])

# Run a Naïve Bayes classification algorithm using the data
library(e1071)
model_NB <- naiveBayes(PROFITABLE~., data=dftrain_NB)

# Create the confusion matrix for validation data and compute the error rate. 
prediction_NB <- predict(model_NB, newdata = dfvalidation_NB[,-21])
table <- table(dfvalidation_NB$PROFITABLE,prediction_NB,dnn=list('actual','predicted'))

(error_rate <- (table[1,2] + table[2,1]) / sum(table))

# Plot the lift chart for the Naïve Bayes model using the validation data
predicted.probability_NB <- predict(model_NB, newdata = dfvalidation_NB[,-21], type="raw")
predicted.probability.NB <- predicted.probability_NB[,2]

# The first column is class 0, the second is class 1
PL <- as.numeric(dfvalidation_NB$PROFITABLE)
prob <- predicted.probability_NB[,2]
df1 <- data.frame(prediction_NB, PL, prob)
#
#
df1S <- df1[order(-prob),]
df1S$Gains <- cumsum(df1S$PL)
plot(df1S$Gains,type="n",main="Lift Chart",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains,col="blue")
abline(0,sum(df1S$PL)/nrow(df1S),lty = 2, col="red")

############## Logistic Regression ##########

# Data Preprocessing
Credit_Log <- data
Credit_Log$AMOUNT_REQUESTED <- as.numeric(gsub(',', '',Credit_Log$AMOUNT_REQUESTED))
Credit_Log$OBS. <- NULL
factor_names <- c(2:3,6,9,19)
Credit_Log[,factor_names] <- lapply(Credit_Log[,factor_names] , factor)
Credit_Log$CREDIT_EXTENDED <- NULL
Credit_Log$PROFITABLE <- ifelse(Credit_Log$NPV > 0, 1, 0)
Credit_Log$NPV <- NULL

# Split data into train and validation
set.seed(12345)
inTrain_Log <- sample(nrow(Credit_Log), 0.7*nrow(Credit_Log))

dftrain_Log <- data.frame(Credit_Log[inTrain_Log,])
dfvalidation_Log <- data.frame(Credit_Log[-inTrain_Log,])

# Run Logistic Regression using the data
Model_Log <- glm(PROFITABLE~., data=dftrain_Log, family="binomial")
summary(Model_Log)

predicted.probability_Log <- predict(Model_Log, type="response", 
                                     newdata=dfvalidation_Log)
# Set cutoff to be 0.5
# If the predicted probability is greater than 0.5, then classify as 1, otherwise 0.
cutoff <- 0.5
Predicted_Log <- ifelse(predicted.probability_Log > cutoff, "1", "0")
table_Log <- table(Predicted_Log)

# Create the confusion matrix for validation data and compute the error rate. 
Actual_Log <- dfvalidation_Log$PROFITABLE
(confusion1 <- table(Actual_Log, Predicted_Log))
prop <- prop.table(confusion1)
cat("Error Rate:", 1- (prop[1,1] + prop[2,2]))

############## Classification Tree ##########

# Data Preprocessing
Credit_Tree <- data
Credit_Tree$PROFITABLE <- ifelse(Credit_Tree$NPV > 0, 1, 0)
factor_names <- c(3:4,7,10,20)
Credit_Tree[,factor_names] <- lapply(Credit_Tree[,factor_names] , factor)
Credit_Tree$CREDIT_EXTENDED <- NULL
Credit_Tree$OBS. <- NULL
Credit_Tree$AMOUNT_REQUESTED <- as.numeric(gsub(',', '',Credit_Tree$AMOUNT_REQUESTED))
Credit_Tree$PROFITABLE <- as.factor(Credit_Tree$PROFITABLE)

# Split the sample into training (70%) and validation (30%) samples 
# with the seed set at 12345. 
set.seed(12345)
inTrain_Tree <- sample(nrow(Credit_Tree), 0.7*nrow(Credit_Tree))
#
dftrain_Tree <- data.frame(Credit_Tree[inTrain_Tree,])
dfvalidation_Tree <- data.frame(Credit_Tree[-inTrain_Tree,])

library(tree)

# Run classification tree using the data
tree.credit=tree(PROFITABLE~.-NPV,dftrain_Tree)
tree.pred=predict(tree.credit, type="class")
set.seed(5)
cv.credit=cv.tree(tree.credit,FUN=prune.misclass)

# Which size minimize the deviation
plot(cv.credit$size,cv.credit$dev,type="b")

# Prune the tree to a best size of 12
prune.credit=prune.misclass(tree.credit,best=12)
plot(prune.credit)
text(prune.credit,pretty=0)

# Create the confusion matrix for validation data and compute the accuracy
predicted.prob_Tree= predict(prune.credit, dfvalidation_Tree, type="vector")[,2]
tree.pred=predict(prune.credit, dfvalidation_Tree, type="class")
(CM.Tree = table(dfvalidation_Tree$PROFITABLE,tree.pred))
(Acc.Tree = (CM.Tree[1,1]+CM.Tree[2,2])/sum(CM.Tree))

############## Model Evaluation ##########

# Evaluate model performance using ROC curves
library(pROC)
par(pty="s")

roc_rose <- plot(roc(dfvalidation$PROFITABLE, predicted.probability), 
                 print.auc = TRUE, col = "blue",legacy.axes=T)

roc_rose <- plot(roc(dfvalidation$PROFITABLE, predicted.probability_Log), print.auc = TRUE, 
                 col = "green", print.auc.y = .44, add = TRUE, legacy.axes=T)

roc_rose <- plot(roc(dfvalidation$PROFITABLE, predicted.probability.NB), print.auc = TRUE, 
                 col = "red", print.auc.y = .38, add = TRUE, legacy.axes=T)

roc_rose <- plot(roc(dfvalidation$PROFITABLE, predicted.prob_Tree), print.auc = TRUE, 
                 col = "purple", print.auc.y = .32, add = TRUE, legacy.axes=T)

legend("bottomright", legend=c("K-Nearest Neighbors", "Naive Bayes", "Logistic Regression", 
                               "Classification Tree"),
       col=c("blue", "red", "green", "purple"), lty=1, cex=0.8, 
       text.font=4, bg='lightblue')

par(pty="m")
