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
library(pROC)
library(gridExtra)



## set working directory ##

setwd("C:/Users/phil0068/DataNow/Home/MASTERS/AI_in_HDS/Summative")

#####  READ IN DATA   ##########

cl1_train <- read.csv("Data/Derived/cl1_m_train_imputed.csv")
cl1_test <- read.csv("Data/Derived/cl1_m_test_imputed.csv")
cl2_train <- read.csv("Data/Derived/cl2_m_train_imputed.csv")
cl2_test <- read.csv("Data/Derived/cl2_m_test_imputed.csv")
cl2_rna_train <- read.csv("Data/Derived/cl2_rna_m_train_imputed.csv")
cl2_rna_test <- read.csv("Data/Derived/cl2_rna_m_test_imputed.csv")

#remove 'X' columns from all datasets
#remove non standardised continuous variables
#remove site_num from cl2 because the factor only includes one level in training set
#remove vary rare variation path_n_num
cl1_train <- cl1_train |> select(-c(X,age))
cl1_test <- cl1_test |> select(-c(X,age))
cl2_train <- cl2_train |> select(-c(X,Age,weight,pack_yrs,site_num,path_m_num))
cl2_test <- cl2_test |> select(-c(X,Age,weight,pack_yrs,site_num,path_m_num))
cl2_rna_train <- cl2_rna_train |> select(-c(X,Age,weight,pack_yrs,site_num,path_m_num))
cl2_rna_test <- cl2_rna_test |> select(-c(X,Age,weight,pack_yrs,site_num,path_m_num))

#Remove ID column from all train and test sets
cl1_train_clean <- cl1_train[, !names(cl1_train) %in% "ID"]
cl1_test_clean <- cl1_test[, !names(cl1_test) %in% "ID"]
cl2_train_clean <- cl2_train[, !names(cl2_train) %in% "ID"]
cl2_test_clean <- cl2_test[, !names(cl2_test) %in% "ID"]
cl2_rna_train_clean <- cl2_rna_train[, !names(cl2_rna_train) %in% "ID"]
cl2_rna_test_clean <- cl2_rna_test[, !names(cl2_rna_test) %in% "ID"]

cl1_train_clean$status_3y <- factor(cl1_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))
cl2_train_clean$status_3y <- factor(cl2_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))
cl2_rna_train_clean$status_3y <- factor(cl2_rna_train_clean$status_3y, 
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
char_cols <- sapply(cl2_rna_train_clean, is.character)
if (any(char_cols)) {
  cat("Converting character columns to factors:", names(char_cols)[char_cols], "\n")
  cl2_rna_train_clean[char_cols] <- lapply(cl2_rna_train_clean[char_cols], as.factor)
}

#Rejoin train and split datasets so that the levels to all factors variables can be known

#Define training control
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

tune_grid <- expand.grid(
  nrounds = c(100, 300),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = c(1, 5),
  subsample = c(0.6, 0.8))

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

#print(cl1_xgb_tune)
#plot(cl1_xgb_tune)

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

#Best parameters
print(cl2_xgb_tune$bestTune)


#Train the model - cl2_rna
set.seed(123)
cl2_rna_xgb_tune <- train(
  status_3y ~ .,
  data = cl2_rna_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0)

#Best parameters
print(cl2_rna_xgb_tune$bestTune)

#save tuned datasets so no need to run again
saveRDS(cl1_xgb_tune, file = "Data/Derived/cl1_xgb_tune.rds")
saveRDS(cl2_xgb_tune, file = "Data/Derived/cl2_xgb_tune.rds")
saveRDS(cl2_rna_xgb_tune, file = "Data/Derived/cl2_rna_xgb_tune.rds")


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

cl2_rna_xbg_model <- train(
  status_3y ~ .,
  data = cl2_rna_train_clean,
  method = "xgbTree",
  trControl = trainControl(
    method = "none", 
    classProbs = TRUE),
  tuneGrid = cl2_rna_xgb_tune$bestTune,  
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


#clinical2 and rna
predictions_prob_cl2_rna <- predict(cl2_rna_xbg_model, 
                                newdata = cl2_rna_test_clean, 
                                type = "prob")[, "Died"]

predictions_class_cl2_rna <- predict(cl2_rna_xbg_model, newdata = cl2_rna_test_clean)

head(data.frame(
  Probability = predictions_prob_cl2_rna,
  Predicted_Class = predictions_class_cl2_rna))


############## Produce outputs and plots for model performance ############


#################
#    ROC AUC    #
#################

#For cl1 model
actual_cl1 <- cl1_test_clean$status_3y 
roc_cl1 <- roc(actual_cl1, predictions_prob_cl1)

roc_df_cl1 <- data.frame(
  specificity = roc_cl1$specificities,
  sensitivity = roc_cl1$sensitivities,
  Model = "Clinical Model 1")

#For cl2 model
actual_cl2 <- cl2_test_clean$status_3y
roc_cl2 <- roc(actual_cl2, predictions_prob_cl2)

roc_df_cl2 <- data.frame(
  specificity = roc_cl2$specificities,
  sensitivity = roc_cl2$sensitivities,
  Model = "Clinical Model 2")

#For cl2_rna model
actual_cl2_rna <- cl2_rna_test_clean$status_3y
roc_cl2_rna <- roc(actual_cl2_rna, predictions_prob_cl2_rna)

roc_df_cl2_rna <- data.frame(
  specificity = roc_cl2_rna$specificities,
  sensitivity = roc_cl2_rna$sensitivities,
  Model = "Clinical & RNA Model 3")

#combine
roc_df <- rbind(roc_df_cl1, roc_df_cl2, roc_df_cl2_rna)

#store AUC values
auc_cl1 <- auc(roc_cl1)
auc_cl2 <- auc(roc_cl2)
auc_cl2_rna <- auc(roc_cl2_rna)

#plot
p1 <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray50", alpha = 0.7) +
  scale_color_manual(values = c("Clinical Model 1" = "#2E86AB", 
                                "Clinical Model 2" = "#A23B72",
                                "Clinical & RNA Model 3" = "red"),
                     labels = c(paste0("Clinical Model 1 (AUC = ", round(auc_cl1, 3), ")"),
                                paste0("Clinical Model 2 (AUC = ", round(auc_cl2, 3), ")"),
                                paste0("Clinical & RNA Model 3 (AUC = ", round(auc_cl2_rna, 3), ")"))) +
  labs(x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)",
       title = "ROC Curves for XGBoost Models",
       color = "Model") +
  theme_minimal() +
  theme(legend.position = c(0.75, 0.25),
        legend.background = element_rect(fill = "white", 
                                         linetype = "solid", 
                                         color = "gray80"),
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10)) +
  coord_equal()

ggsave("Output/cl1_and_cl2_and_cl2rna_ROC.png", plot = p1)


#################
#    Brier      #
#################

brier_cl1 <- mean((predictions_prob_cl1 - actual_cl1)^2)

brier_cl2 <- mean((predictions_prob_cl2 - actual_cl2)^2)

brier_cl2_rna <- mean((predictions_prob_cl2_rna - actual_cl2_rna)^2)

cat("Brier Score for Clinical Model 1:", round(brier_cl1, 4), "\n")
cat("Brier Score for Clinical Model 2:", round(brier_cl2, 4), "\n")
cat("Brier Score for Clinical & RNA Model 3:", round(brier_cl2_rna, 4), "\n")


#=0.224, 0.244, 0.213 'moderate'

#####################
# Calibration curve #
#####################

df_cl1 <- data.frame(
  pred = predictions_prob_cl1,
  actual = actual_cl1,
  Model = "Clinical Model 1")

df_cl2 <- data.frame(
  pred = predictions_prob_cl2,
  actual = actual_cl2,
  Model = "Clinical Model 2")

df_cl2_rna <- data.frame(
  pred = predictions_prob_cl2_rna,
  actual = actual_cl2_rna,
  Model = "Clinical & RNA Model 3")

combined_df <- rbind(df_cl1, df_cl2, df_cl2_rna)

p2 <- ggplot(combined_df, aes(x = pred, y = actual, color = Model)) +
  # Add loess smooth line with confidence interval
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2, size = 1.2) +
  # Add ideal calibration line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "black", size = 1) +
  # Add rug plot to show distribution
  geom_rug(data = df_cl1, aes(x = pred), sides = "b", alpha = 0.1, 
           color = "#2E86AB", position = "jitter") +
  geom_rug(data = df_cl2, aes(x = pred), sides = "t", alpha = 0.1, 
           color = "#A23B72", position = "jitter") +
  geom_rug(data = df_cl2_rna, aes(x = pred), sides = "b", alpha = 0.1, 
           color = "red", position = "jitter") +
  scale_color_manual(values = c("Clinical Model 1" = "#2E86AB", 
                                "Clinical Model 2" = "#A23B72",
                                "Clinical & RNA Model 3" = "red")) +
  labs(x = "Predicted Probability of Death",
       y = "Observed Proportion of Death",
       title = "Calibration Curves with LOESS Smoothing",
       subtitle = "Shaded areas represent 95% confidence intervals") +
  theme_minimal() +
  theme(legend.position = c(0.2, 0.8),
        legend.background = element_rect(fill = "white", linetype = "solid"),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5)) +
  coord_cartesian(xlim = c(0, 0.8), ylim = c(0, 1))


ggsave("Output/cl1_and_cl2_and_cl2rna_calib.png", plot = p2)




#####################################################
#
#     addressing class imbalance coding       
#
#####################################################

### Approach 1: XGBoost scale_pos_weight

#calculate class imbalance ratio for cl1
cl1_class_counts <- table(cl1_train_clean$status_3y)
cl1_scale_pos_weight <- cl1_class_counts["Alive"] / cl1_class_counts["Died"]
cat("cl1 - Alive:", cl1_class_counts["Alive"], 
    "Died:", cl1_class_counts["Died"], 
    "scale_pos_weight:", cl1_scale_pos_weight, "\n")

#calculate for cl2
cl2_class_counts <- table(cl2_train_clean$status_3y)
cl2_scale_pos_weight <- cl2_class_counts["Alive"] / cl2_class_counts["Died"]
cat("cl2 - Alive:", cl2_class_counts["Alive"], 
    "Died:", cl2_class_counts["Died"], 
    "scale_pos_weight:", cl2_scale_pos_weight, "\n")

#calcualte for cl2_rna
cl2_rna_class_counts <- table(cl2_rna_train_clean$status_3y)
cl2_rna_scale_pos_weight <- cl2_rna_class_counts["Alive"] / cl2_rna_class_counts["Died"]
cat("cl2_rna - Alive:", cl2_rna_class_counts["Alive"], 
    "Died:", cl2_rna_class_counts["Died"], 
    "scale_pos_weight:", cl2_rna_scale_pos_weight, "\n")

#tuning grid with scale_pos_weight
tune_grid_weighted <- expand.grid(
  nrounds = c(100, 300),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = c(1, 5),
  subsample = c(0.6, 0.8))

#train cl1 with scale_pos_weight
set.seed(123)
cl1_xgb_tune_weighted <- train(
  status_3y ~ .,
  data = cl1_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid_weighted,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0,
  scale_pos_weight = cl1_scale_pos_weight)

#train cl2 with scale_pos_weight
set.seed(123)
cl2_xgb_tune_weighted <- train(
  status_3y ~ .,
  data = cl2_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid_weighted,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0,
  scale_pos_weight = cl2_scale_pos_weight)

#train cl2_rna with scale_pos_weight
set.seed(123)
cl2_rna_xgb_tune_weighted <- train(
  status_3y ~ .,
  data = cl2_rna_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid_weighted,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0,
  scale_pos_weight = cl2_rna_scale_pos_weight)

#save tuned models so don't need to run again
saveRDS(cl1_xgb_tune_weighted, file = "Data/Derived/cl1_xgb_tune_weighted.rds")
saveRDS(cl2_xgb_tune_weighted, file = "Data/Derived/cl2_xgb_tune_weighted.rds")
saveRDS(cl2_rna_xgb_tune_weighted, file = "Data/Derived/cl2_rna_xgb_tune_weighted.rds")


#best models
cl1_xgb_best_weighted <- train(
  status_3y ~ .,
  data = cl1_train_clean,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl1_xgb_tune_weighted$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0,
  scale_pos_weight = cl1_scale_pos_weight)

cl2_xgb_best_weighted <- train(
  status_3y ~ .,
  data = cl2_train_clean,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl2_xgb_tune_weighted$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0,
  scale_pos_weight = cl2_scale_pos_weight)

cl2_rna_xgb_best_weighted <- train(
  status_3y ~ .,
  data = cl2_rna_train_clean,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl2_rna_xgb_tune_weighted$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0,
  scale_pos_weight = cl2_rna_scale_pos_weight)

#compare performance
cat("Without weighting - ROC:", max(cl1_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl1_xgb_tune_weighted$results$ROC), "\n")

cat("Without weighting - ROC:", max(cl2_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl2_xgb_tune_weighted$results$ROC), "\n")

cat("Without weighting - ROC:", max(cl2_rna_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl2_rna_xgb_tune_weighted$results$ROC), "\n")

#compare predictions on test set
predictions_weighted <- predict(cl1_xgb_tune_weighted, 
                                newdata = cl1_test_clean, 
                                type = "prob")[, "Died"]

pred_cl1_original <- predict(cl1_xbg_model, newdata = cl1_test_clean, type = "prob")[, "Died"]
pred_cl1_weighted <- predict(cl1_xgb_best_weighted, newdata = cl1_test_clean, type = "prob")[, "Died"]

pred_cl2_original <- predict(cl2_xbg_model, newdata = cl2_test_clean, type = "prob")[, "Died"]
pred_cl2_weighted <- predict(cl2_xgb_best_weighted, newdata = cl2_test_clean, type = "prob")[, "Died"]

pred_cl2_rna_original <- predict(cl2_rna_xbg_model, newdata = cl2_rna_test_clean, type = "prob")[, "Died"]
pred_cl2_rna_weighted <- predict(cl2_rna_xgb_best_weighted, newdata = cl2_rna_test_clean, type = "prob")[, "Died"]

roc_cl1_original <- roc(cl1_test_clean$status_3y, pred_cl1_original)
roc_cl1_weighted <- roc(cl1_test_clean$status_3y, pred_cl1_weighted)
roc_cl2_original <- roc(cl2_test_clean$status_3y, pred_cl2_original)
roc_cl2_weighted <- roc(cl2_test_clean$status_3y, pred_cl2_weighted)
roc_cl2_rna_original <- roc(cl2_rna_test_clean$status_3y, pred_cl2_rna_original)
roc_cl2_rna_weighted <- roc(cl2_rna_test_clean$status_3y, pred_cl2_rna_weighted)


plot_cl1 <- ggroc(list(
  Original = roc_cl1_original,Weighted = roc_cl1_weighted), size = 1.2, alpha = 0.8) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
               linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Original" = "#2E86AB",
               "Weighted" = "#A23B72"),
    labels = c(
      paste0("Original (AUC = ", round(auc(roc_cl1_original), 3), ")"),
      paste0("Weighted (AUC = ", round(auc(roc_cl1_weighted), 3), ")"))) +
  labs(
    x = "1 - Specificity",
    y = "Sensitivity",
    title = "Clinical Model 1: Original vs Weighted XGBoost",
    color = "Model") +
  theme_minimal() +
  theme(
    legend.position = c(0.8, 0.2),
    legend.background = element_rect(
      fill = "white",
      linetype = "solid",
      color = "gray80"),
    plot.title = element_text(hjust = 0.5, face = "bold"))

plot_cl2 <- ggroc(list(
  Original = roc_cl2_original,Weighted = roc_cl2_weighted),size = 1.2, alpha = 0.8) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
               linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Original" = "#2E86AB",
               "Weighted" = "#A23B72"),
    labels = c(
      paste0("Original (AUC = ", round(auc(roc_cl2_original), 3), ")"),
      paste0("Weighted (AUC = ", round(auc(roc_cl2_weighted), 3), ")"))) +
  labs(
    x = "1 - Specificity",
    y = "Sensitivity",
    title = "Clinical Model 2: Original vs Weighted XGBoost",
    color = "Model") +
  theme_minimal() +
  theme(
    legend.position = c(0.8, 0.2),
    legend.background = element_rect(
      fill = "white",
      linetype = "solid",
      color = "gray80"),
    plot.title = element_text(hjust = 0.5, face = "bold"))

plot_cl2_rna <- ggroc(list(
  Original = roc_cl2_rna_original,Weighted = roc_cl2_rna_weighted),size = 1.2, alpha = 0.8) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
               linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Original" = "#2E86AB",
               "Weighted" = "#A23B72"),
    labels = c(
      paste0("Original (AUC = ", round(auc(roc_cl2_rna_original), 3), ")"),
      paste0("Weighted (AUC = ", round(auc(roc_cl2_rna_weighted), 3), ")"))) +
  labs(
    x = "1 - Specificity",
    y = "Sensitivity",
    title = "Clinical Model 2: Original vs Weighted XGBoost",
    color = "Model") +
  theme_minimal() +
  theme(
    legend.position = c(0.8, 0.2),
    legend.background = element_rect(
      fill = "white",
      linetype = "solid",
      color = "gray80"),
    plot.title = element_text(hjust = 0.5, face = "bold"))

grid.arrange(plot_cl1, plot_cl2, ncol = 2)

ggsave("Output/cl1_xgb_oversampling_AUG.png", plot = plot_cl1)
ggsave("Output/cl2_xgb_oversampling_AUG.png", plot = plot_cl2)
ggsave("Output/cl2_rna_xgb_oversampling_AUG.png", plot = plot_cl2_rna)


###### Approach 2: Random sampling


#Upsample cl1
set.seed(123)
cl1_train_upsampled <- upSample(
  x = cl1_train_clean[, !names(cl1_train_clean) %in% "status_3y"],
  y = cl1_train_clean$status_3y,
  yname = "status_3y")

#Upsample cl2
set.seed(123)
cl2_train_upsampled <- upSample(
  x = cl2_train_clean[, !names(cl2_train_clean) %in% "status_3y"],
  y = cl2_train_clean$status_3y,
  yname = "status_3y")

# Check balance
table(cl1_train_upsampled$status_3y)
table(cl2_train_upsampled$status_3y)

#train models on upsampled data
set.seed(123)
cl1_xgb_upsampled <- train(
  status_3y ~ .,
  data = cl1_train_upsampled,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE,
    classProbs = TRUE,
    summaryFunction = twoClassSummary),
  tuneGrid = tune_grid,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0)

# Train CL2 model on upsampled data
set.seed(123)
cl2_xgb_upsampled <- train(
  status_3y ~ .,
  data = cl2_train_upsampled,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE,
    classProbs = TRUE,
    summaryFunction = twoClassSummary),
  tuneGrid = tune_grid,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0)

cl1_best_upsampled <- train(
  status_3y ~ .,
  data = cl1_train_upsampled,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl1_xgb_upsampled$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0)

cl2_best_upsampled <- train(
  status_3y ~ .,
  data = cl2_train_upsampled,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl2_xgb_upsampled$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0)

pred_cl1_original <- predict(cl1_xbg_model, newdata = cl1_test_clean, type = "prob")[, "Died"]
pred_cl1_upsampled <- predict(cl1_best_upsampled, newdata = cl1_test_clean, type = "prob")[, "Died"]

pred_cl2_original <- predict(cl2_xbg_model, newdata = cl2_test_clean, type = "prob")[, "Died"]
pred_cl2_upsampled <- predict(cl2_best_upsampled, newdata = cl2_test_clean, type = "prob")[, "Died"]

roc_cl1_original <- roc(cl1_test_clean$status_3y, pred_cl1_original)
roc_cl1_upsampled <- roc(cl1_test_clean$status_3y, pred_cl1_upsampled)

roc_cl2_original <- roc(cl2_test_clean$status_3y, pred_cl2_original)
roc_cl2_upsampled <- roc(cl2_test_clean$status_3y, pred_cl2_upsampled)

cat("CL1 Original AUC:", round(auc(roc_cl1_original), 4), "\n")
cat("CL1 Upsampled AUC:", round(auc(roc_cl1_upsampled), 4), "\n")
cat("CL2 Original AUC:", round(auc(roc_cl2_original), 4), "\n")
cat("CL2 Upsampled AUC:", round(auc(roc_cl2_upsampled), 4), "\n")

plot_cl1 <- ggroc(list(
  Original = roc_cl1_original,Weighted = roc_cl1_upsampled), size = 1.2, alpha = 0.8) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
               linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Original" = "#2E86AB",
               "Weighted" = "#A23B72"),
    labels = c(
      paste0("Original (AUC = ", round(auc(roc_cl1_original), 3), ")"),
      paste0("Weighted (AUC = ", round(auc(roc_cl1_upsampled), 3), ")"))) +
  labs(
    x = "1 - Specificity",
    y = "Sensitivity",
    title = "Clinical Model 1: Original vs Upsampled XGBoost",
    color = "Model") +
  theme_minimal() +
  theme(
    legend.position = c(0.8, 0.2),
    legend.background = element_rect(
      fill = "white",
      linetype = "solid",
      color = "gray80"),
    plot.title = element_text(hjust = 0.5, face = "bold"))

plot_cl2 <- ggroc(list(
  Original = roc_cl2_original,Weighted = roc_cl2_upsampled), size = 1.2, alpha = 0.8) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
               linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("Original" = "#2E86AB",
               "Weighted" = "#A23B72"),
    labels = c(
      paste0("Original (AUC = ", round(auc(roc_cl2_original), 3), ")"),
      paste0("Weighted (AUC = ", round(auc(roc_cl2_upsampled), 3), ")"))) +
  labs(
    x = "1 - Specificity",
    y = "Sensitivity",
    title = "Clinical Model 2: Original vs Upsampled XGBoost",
    color = "Model") +
  theme_minimal() +
  theme(
    legend.position = c(0.8, 0.2),
    legend.background = element_rect(
      fill = "white",
      linetype = "solid",
      color = "gray80"),
    plot.title = element_text(hjust = 0.5, face = "bold"))

grid.arrange(plot_cl1, plot_cl2, ncol = 2)

ggsave("Output/cl1_xgb_upsampling_AUG.png", plot = plot_cl1)
ggsave("Output/cl2_xgb_upsampling_AUG.png", plot = plot_cl2)


roc_test_cl1 <- roc.test(roc_cl1_original, roc_cl1_upsampled, method = "delong")
roc_test_cl2 <- roc.test(roc_cl2_original, roc_cl2_upsampled, method = "delong")

cat("CL1: p-value =", roc_test_cl1$p.value, 
    ifelse(roc_test_cl1$p.value < 0.05, " - Significant difference", " - No significant difference"), "\n")
cat("CL2: p-value =", roc_test_cl2$p.value,
    ifelse(roc_test_cl2$p.value < 0.05, " - Significant difference", " - No significant difference"), "\n")

