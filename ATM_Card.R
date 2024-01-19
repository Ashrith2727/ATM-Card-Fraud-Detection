#DataSet access: https://drive.google.com/file/d/1K47Kf8Ack2ILD2fIsLHYbWvkYwBEAOK3/view?usp=sharing

# Importing Libraries
library(ranger)
library(caret)
library(data.table)

# Reading Data
creditcard_data <- read.csv("creditcard.csv")

# Data Exploration
dim(creditcard_data)
head(creditcard_data, 6)
tail(creditcard_data, 6)
table(creditcard_data$Class)

# Data Manipulation
creditcard_data$Amount = scale(creditcard_data$Amount)
NewData = creditcard_data[, -c(1)]

# Data Modelling
set.seed(123)
data_sample = sample.split(NewData$Class, SplitRatio = 0.80)
train_data = subset(NewData, data_sample == TRUE)
test_data = subset(NewData, data_sample == FALSE)

# Fitting Logistic Regression Model
Logistic_Model = glm(Class ~ ., test_data, family = binomial())
summary(Logistic_Model)

# Fitting Decision Tree Model
decisionTree_model <- rpart(Class ~ ., creditcard_data, method = 'class')

# Artificial Neural Network
ANN_model = neuralnet(Class ~ ., train_data, linear.output = FALSE)

# Gradient Boosting (GBM)
model_gbm <- gbm(Class ~ ., distribution = "bernoulli", data = rbind(train_data, test_data), n.trees = 500, interaction.depth = 3, n.minobsinnode = 100, shrinkage = 0.01, bag.fraction = 0.5, train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data)))

# Plot and calculate AUC on test data
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)
