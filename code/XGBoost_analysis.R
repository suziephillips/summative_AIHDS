#############################################################################
#                                                                           #
# Program Name:  XGBoost Analysis                                           #
#                                                                           #
# Outputs:      
#
#                                                                           #
#############################################################################


###############################
##      LOAD PACKAGES       ##
##############################
library(tidyverse)
library(ggplot2)
library(lattice)
library(xgboost)
library(caret)
library(dplyr)

## set working directory ##

setwd("C:/Users/phil0068/DataNow/Home/MASTERS/AI_in_HDS/Summative")

#####  READ IN DATA   ##########

cl1_train <- read.csv("Data/Derived/cl1_train.csv")
cl1_test <- read.csv("Data/Derived/cl1_test.csv")
cl2_train <- read.csv("Data/Derived/cl2_train.csv")
cl2_test <- read.csv("Data/Derived/cl2_test.csv")
#cl2_rna_train <- read.csv("Data/Derived/cl2_rna_train.csv")
#cl2_rna_test <- read.csv("Data/Derived/cl2_rna_test.csv")

#remove 'X' columns from all datasets
#remove the time-to-event and cencoring data from the datasets as will be doing a binary target
#remove non standardised continuous variables
#remove site_num from cl2 because the factor only includes one level in training set
#remove dates
cl1_train <- cl1_train |> select(-c(X,surv,cens,surv_3y,cens_3y,age))
cl1_test <- cl1_test |> select(-c(X,surv,cens,surv_3y,cens_3y,age))
cl2_train <- cl2_train |> select(-c(X,surv,cens,surv_3y,cens_3y,Age,weight,pack_yrs,site_num,Date.of.Recurrence,PET.Date))
cl2_test <- cl2_test |> select(-c(X,surv,cens,surv_3y,cens_3y,Age,weight,pack_yrs,site_num,Date.of.Recurrence,PET.Date))

#Remove ID column from all train and test sets
cl1_train_clean <- cl1_train[, !names(cl1_train) %in% "ID"]
cl1_test_clean <- cl1_test[, !names(cl1_test) %in% "ID"]
cl2_train_clean <- cl2_train[, !names(cl1_train) %in% "ID"]
cl2_test_clean <- cl2_test[, !names(cl1_test) %in% "ID"]

#Remove rows with missing target in training and test
cl1_train_clean <- cl1_train_clean[!is.na(cl1_train_clean$status_3y), ]
cl2_train_clean <- cl2_train_clean[!is.na(cl2_train_clean$status_3y), ]
cl1_test_clean <- cl1_test_clean[!is.na(cl1_test_clean$status_3y), ]
cl2_test_clean <- cl2_test_clean[!is.na(cl2_test_clean$status_3y), ]

cl1_train_clean$status_3y <- factor(cl1_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))
cl2_train_clean$status_3y <- factor(cl2_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))

#Convert character columns to factors
char_cols <- sapply(cl1_train_clean, is.character)
if (any(char_cols)) {
  cat("Converting character columns to factors:", names(char_cols)[char_cols], "\n")
  cl1_train_clean[char_cols] <- lapply(cl1_train_clean[char_cols], as.factor)
}
char_cols <- sapply(cl2_train_clean, is.character)
if (any(char_cols)) {
  cat("Converting character columns to factors:", names(char_cols)[char_cols], "\n")
  cl2_train_clean[char_cols] <- lapply(cl2_train_clean[char_cols], as.factor)
}

#Define training control
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

# Define tuning grid
tune_grid <- expand.grid(
  nrounds = 100,
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1, 0.3),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8)

#Train the model - cl1
set.seed(123)
cl1_xgb_tune <- train(
  status_3y ~ .,
  data = cl1_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0)

#Results
print(cl1_xgb_tune)
plot(cl1_xgb_tune)

#Best parameters
print(cl1_xgb_tune$bestTune)

#Train the model - cl2
set.seed(123)
cl2_xgb_tune <- train(
  status_3y ~ .,
  data = cl2_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0)

#Results
print(cl2_xgb_tune)
plot(cl2_xgb_tune)

#Best parameters
print(cl2_xgb_tune$bestTune)


#Train the model - cl2_rna



########## Train XGBoost model with best parameters ##########


cl1_xbg_model <- train(
  status_3y ~ .,
  data = cl1_train_clean,
  method = "xgbTree",
  trControl = trainControl(
    method = "none", 
    classProbs = TRUE),
  tuneGrid = cl1_xgb_tune$bestTune,  
  na.action = na.pass,
  nthread = 4,
  verbose = 1)

cl2_xbg_model <- train(
  status_3y ~ .,
  data = cl2_train_clean,
  method = "xgbTree",
  trControl = trainControl(
    method = "none", 
    classProbs = TRUE),
  tuneGrid = cl2_xgb_tune$bestTune,  
  na.action = na.pass,
  nthread = 4,
  verbose = 1)

########## make predcitions using test set ##############

##clinical1 
predictions_prob_cl1 <- predict(cl1_xbg_model, 
                            newdata = cl1_test_clean, 
                            type = "prob")[, "Died"]

predictions_class_cl1 <- predict(cl1_xbg_model, newdata = cl1_test_clean)

head(data.frame(
  Probability = predictions_prob_cl1,
  Predicted_Class = predictions_class_cl1))

#clinical2
predictions_prob_cl2 <- predict(cl2_xbg_model, 
                                newdata = cl2_test_clean, 
                                type = "prob")[, "Died"]

predictions_class_cl2 <- predict(cl2_xbg_model, newdata = cl2_test_clean)

head(data.frame(
  Probability = predictions_prob_cl2,
  Predicted_Class = predictions_class_cl2))

############## Produce outputs and plots for model performance ############



library(pROC)
library(tidyr)
library(gridExtra)


# ============================================
# 1. PREPARE DATA FOR EVALUATION
# ============================================

cat("Rows in cl1_test_clean:", nrow(cl1_test_clean), "\n")
cat("Rows in predictions:", length(predictions_prob_cl1), "\n")
cat("Difference:", nrow(cl1_test_clean) - length(predictions_prob_cl1), "rows missing from predictions\n")

# Method 1: If your predictions align with cl1_test_clean rows
# Create test_with_target by combining predictions with actual labels
test_with_target <- cl1_test_clean

predictions_prob_cl1$pred_class <- predict(cl1_xbg_model, 
                                       newdata = predictions_prob_cl1)

# Extract actual and predicted values
actual <- as.numeric(predictions_prob_cl1$status_3y == "Died")
pred_prob <- predictions_prob_cl1$pred_prob




















# ============================================
# 2. ROC-AUC PLOT
# ============================================

# Calculate ROC curve
roc_obj <- roc(actual, pred_prob)
auc_value <- auc(roc_obj)
auc_ci <- ci.auc(roc_obj)

# Create ROC data frame for ggplot
roc_df <- data.frame(
  sensitivity = roc_obj$sensitivities,
  specificity = roc_obj$specificities,
  fpr = 1 - roc_obj$specificities
)

# ROC Plot
roc_plot <- ggplot(roc_df, aes(x = fpr, y = sensitivity)) +
  geom_line(color = "#2E86AB", size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50", alpha = 0.7) +
  labs(
    title = "ROC Curve",
    subtitle = paste("AUC =", round(auc_value, 3), 
                     " (95% CI:", round(auc_ci[1], 3), "-", round(auc_ci[3], 3), ")"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "gray70", size = 0.5)
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2))

print(roc_plot)




