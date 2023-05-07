# Clear workspace
rm(list = ls())


# Set working directory
setwd("~/Desktop/UMiami/stats/Stroke_predictions")

#import data
library(readr)
df <- read_csv("healthcare-dataset-stroke-data.csv", 
                col_types = cols(work_type = col_skip(), id = col_skip()), #skips columns
                                        na = c("N/A", "Unknown")) 

#Clean the data
df = na.omit(df) 
df$gender <- ifelse(df$gender == "Male", 1,0)
df$ever_married <- ifelse(df$ever_married == "Yes", 1,0)
df$Residence_type <- ifelse(df$Residence_type == "Urban", 1,0)
df$smoking_status <- ifelse(df$smoking_status == "never smoked", 0,1)


#Correlation between different attributes
library(ggplot2)
library(reshape2)

cor(df)

cormat = round(cor(df),2)
melted_cormat = melt(cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile()

#Split dataset into training and testing (80/20).
set.seed(150)  # set a seed for reproducibility
 
#####Stratified sampling to evenly distribute 0 and 1
library(ROSE)

table(df$stroke)

# generate new balanced data by ROSE
df_rose<- ROSE(stroke ~ ., data=df, seed=123)$data
# check (im)balance of new data
table(df_rose$stroke)

training_index <- sample(1:nrow(df_rose), size = 0.8 * nrow(df_rose))
training_set <- df_rose[training_index, ] # 2740 samples
testing_set <- df_rose[-training_index, ] # 686 samples

dim(training_set)
dim(testing_set)

#Model 1:Logistic Regression Model
glm.fit1 = glm(stroke~.,data = training_set, family = "binomial")
summary(glm.fit1)

glm.probs = predict(glm.fit1,newdata = testing_set, type="response")
# Convert predicted probabilities to binary predictions
test_predictions_binary <- ifelse(glm.probs >= 0.5, 1, 0)
# Create a confusion matrix
glm_confusion_matrix <- table(test_predictions_binary, testing_set$stroke)
# Calculate the test error
glm_precision <- glm_confusion_matrix[2,2] / (glm_confusion_matrix[2,2] + glm_confusion_matrix[1,2])
glm_recall <- glm_confusion_matrix[2,2] / (glm_confusion_matrix[2,2] + glm_confusion_matrix[2,1])
glm_F1_score <- 2 * (glm_precision * glm_recall) / (glm_precision + glm_recall)
glm_accuracy <- sum(diag(glm_confusion_matrix)) / sum(glm_confusion_matrix)


# Visualize the confusion matrix
dimnames(glm_confusion_matrix) <- list(test_predictions_binary = 
                                      c("Negative", "Positive"), class = c("Negative", "Positive"))

#Evalutation Metrics
glm_confusion_matrix
glm_precision
glm_recall
glm_F1_score
glm_accuracy



########
# Tree #
########

library(ISLR2) #linear Regression
library(tree) #for random tree generator
library(randomForest) #For bagging and random forests
library(gbm) #for boosting
library(caret)

# Train using training set
# tree(): Fit a classification tree 
tree.model= tree(as.factor(stroke)~., data = training_set)
summary(tree.model)
plot(tree.model)
text(tree.model, pretty = 0, cex=0.75)
tree.model

## Validation set
tree.pred=predict(tree.model,testing_set,type="class")

# Confusion matrix
tree_confusion_matrix <- table(tree.pred,testing_set$stroke)
tree_confusion_matrix

# Accuracy on test set
tree_precision <- tree_confusion_matrix[2,2] / (tree_confusion_matrix[2,2] + tree_confusion_matrix[1,2])
tree_recall <- tree_confusion_matrix[2,2] / (tree_confusion_matrix[2,2] + tree_confusion_matrix[2,1])
tree_F1_score <- 2 * (tree_precision * tree_recall) / (tree_precision + tree_recall)
tree_accuracy <- sum(diag(tree_confusion_matrix)) / sum(tree_confusion_matrix)

#Evalutation Metrics
tree_confusion_matrix
tree_precision
tree_recall
tree_F1_score
tree_accuracy

#Run Cross-Validation on Tree

tree_cv=cv.tree(tree.model,FUN=prune.misclass)

tree_cv$size
tree_cv$dev
tree_cv$k

# Plot the error rate as a function of both size and k
par(mfrow=c(1,2))
plot(tree_cv$size, tree_cv$dev, type="b")
plot(tree_cv$k, tree_cv$dev, type="b")

# prune.misclass(): prune the tree based on the CV error
prune_tree=prune.misclass(tree.model,best=9)

# Plot resulting decision tree
plot(prune_tree)
text(prune_tree, pretty=0, cex=0.75)


prune.pred=predict(prune_tree,testing_set,type="class")
prune_confusion_matrix <- table(prune.pred,testing_set$stroke)
prune_confusion_matrix


###############
#Random Forest#
###############

# m=sqrt()
n <- ncol(df)
m <- sqrt(n)

#train
rf.model=randomForest(as.factor(stroke)~.,data=training_set,mtry=m,importance =T)
rf.importance <- importance(rf.model)
rf.importance


#predict
yhat.rf = predict(rf.model,newdata=testing_set)
plot(yhat.rf, testing_set$stroke)
testing_set_stroke <- factor(testing_set$stroke)


rf_confusion_matrix <- confusionMatrix(table(yhat.rf, testing_set$stroke, dnn = c("Predicted", "Actual")), positive = "1")
rf_summary <- rf_confusion_matrix$byClass
rf_precision <- rf_summary["Precision"]
rf_recall <- rf_summary["Recall"]
rf_f1_score <- rf_summary["F1"]
rf_accuracy <- rf_confusion_matrix$overall["Accuracy"]

rf_confusion_matrix
rf_precision
rf_recall
rf_f1_score
rf_accuracy

#####
#SVM#
#####

library(e1071)

svc_model=svm(training_set$stroke~., data=training_set, kernel="linear", cost=10, scale=TRUE)
summary(svc_model)

svc_pred=predict(svc_model, testing_set)
svc_pred <- factor(ifelse(svc_pred > 0.5, "1", "0"), levels = c("0", "1"))
svc_cm <- confusionMatrix(svc_pred, as.factor(testing_set$stroke), positive = "1")
print(svc_cm)

svc_summary <- svc_cm$byClass
svc_precision <- svc_summary["Precision"]
svc_recall <- svc_summary["Recall"]
svc_f1_score <- svc_summary["F1"]
svc_accuracy <- svc_cm$overall["Accuracy"]

svc_cm
svc_precision
svc_recall
svc_f1_score
svc_accuracy

#CV SVC


svc.tune.out <- tune(svm, stroke~., data = training_set, 
                     kernel = "linear", 
                     ranges =list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), decision.values=T)

summary(svc.tune.out)

svc.bestmod=svc.tune.out$best.model
summary(svc.bestmod)

svc_pred2=predict(svc.bestmod, testing_set)
svc_pred2 = factor(ifelse(svc_pred2 > 0.5, "1", "0"), levels = c("0", "1"))
svc_cm2 <- confusionMatrix(svc_pred2, as.factor(testing_set$stroke), positive = "1")
print(svc_cm)

#SVM
svm_model=svm(training_set$stroke~., data=training_set, kernel="radial", gamma=1, cost=1)
summary(svm_model)

svm_pred=predict(svm_model, testing_set)
svm_pred <- factor(ifelse(svm_pred > 0.5, "1", "0"), levels = c("0", "1"))
svm_cm <- confusionMatrix(svm_pred, as.factor(testing_set$stroke), positive = "1")
print(svm_cm)

#CV
svm.tune.out <- tune(svm, stroke~., data = training_set, 
                     kernel = "radial", ranges=list(cost=c(0.1,1,10,100,1000),
                                                    gamma=c(0.5,1,2,3,4) ), decision.values=T)
summary(svm.tune.out)

svm.bestmod=svm.tune.out$best.model
summary(svm.bestmod)

svm_pred2=predict(svm.bestmod, testing_set)
svm_pred2 <- factor(ifelse(svm_pred2 > 0.5, "1", "0"), levels = c("0", "1"))
svm_cm2 <- confusionMatrix(svm_pred2, as.factor(testing_set$stroke), positive = "1")
print(svm_cm2)

#######Stacking


