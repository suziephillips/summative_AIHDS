#############################################################################
#                                                                           #
# Program Name:  Neural Network Analysis                                           #
#                                                                           #
#############################################################################


###############################
##      LOAD PACKAGES       ##
##############################
library(ggplot2)
library(glmnet)
library(pROC)
library(caret)
library(keras)
library(tensorflow)
library(dplyr)


## set working directory ##

setwd("C:/Users/phil0068/DataNow/Home/MASTERS/AI_in_HDS/Summative")

#####  READ IN DATA   ##########

cl1_train <- read.csv("Data/Derived/cl1_m_train_imputed.csv")
cl1_test <- read.csv("Data/Derived/cl1_m_test_imputed.csv")
cl2_train <- read.csv("Data/Derived/cl2_m_train_imputed.csv")
cl2_test <- read.csv("Data/Derived/cl2_m_test_imputed.csv")
cl2_rna_train <- read.csv("Data/Derived/cl2_rna_m_train_imputed.csv")
cl2_rna_test <- read.csv("Data/Derived/cl2_rna_m_test_imputed.csv")
cl1_img_train <- read.csv("Data/Derived/cl1_img_scl_train_imputed")
cl1_img_test <- read.csv("Data/Derived/cl1_img_scl_test_imputed")

#scale data to prepare for neural network.

#•	5fold cross validation with lasso cox to get feature selection options, for the training split. Then use selection which maximised c-index on the validation split. ensure this is average validation C-index across folds, not a single fold.
#•	Using same folds, tune DeepSurv hyperparameters.
#•	Fit DeepSurv on 4 datasets
#•	Output AUC, Brier score and calibration curves
#•	Run PFI/ICE/PDP to be able to interpret the neural network

# Load required libraries

# Set seed for reproducibility
set.seed(123)

# Assuming your data is split into train_data and test_data
# train_data and test_data are dataframes with 'status_3y' as binary outcome

# Prepare training data
cl1_x_train <- as.matrix(cl1_train[, !names(cl1_train) %in% "status_3y"])
cl1_y_train <- cl1_train$status_3y


# If you have categorical variables, create dummy variables first:
# dummy_model <- dummyVars(~ ., data = train_data[, !names(train_data) %in% "status_3y"])
# x_train <- predict(dummy_model, newdata = train_data)

# 5-fold CV Lasso Logistic
cv_lasso <- cv.glmnet(
  x = cl1_x_train,
  y = cl1_y_train,
  family = "binomial",
  alpha = 1,                    # Lasso
  type.measure = "auc",         # Maximize AUC
  nfolds = 5,
  parallel = FALSE,
  seed = 123)

best_lambda <- cv_lasso$lambda.1se

# Extract selected features
coef_matrix <- as.matrix(coef(cv_lasso, s = best_lambda))
selected_features <- rownames(coef_matrix)[coef_matrix[, 1] != 0]
selected_features <- selected_features[selected_features != "(Intercept)"]

cat("Number of features selected:", length(selected_features), "\n")
cat("Selected features:\n", paste(selected_features[1:min(10, length(selected_features))], collapse = ", "), "\n")

# Create reduced datasets with only selected features
x_train_selected <- x_train[, selected_features, drop = FALSE]
x_test <- as.matrix(test_data[, names(test_data) %in% selected_features])
y_test <- test_data$status_3y

# Verify test set has same features
cat("\nTraining features:", ncol(x_train_selected))
cat("\nTest features:", ncol(x_test))
















