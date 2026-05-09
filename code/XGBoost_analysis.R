#############################################################################
#                                                                           #
# Program Name:  XGBoost Analysis                                           #
#                                                                           #
#  Outputs:  xgboost_roc_curves.png, xgboost_cal_curves_smooth.png          #
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
cl1_img_train <- read.csv("Data/Derived/cl1_img_train_imputed")
cl1_img_test <- read.csv("Data/Derived/cl1_img_test_imputed")

#data used to run random forest has not been scaled as this is naturally done as part of the model.
#when doing NN data will need to be scaled and normalised.

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
cl1_img_train <- cl1_img_train |> select(-c(X,age))
cl1_img_test <- cl1_img_test |> select(-c(X,age))

#Remove ID column from all train and test sets
cl1_train_clean <- cl1_train[, !names(cl1_train) %in% "ID"]
cl1_test_clean <- cl1_test[, !names(cl1_test) %in% "ID"]
cl2_train_clean <- cl2_train[, !names(cl2_train) %in% "ID"]
cl2_test_clean <- cl2_test[, !names(cl2_test) %in% "ID"]
cl2_rna_train_clean <- cl2_rna_train[, !names(cl2_rna_train) %in% "ID"]
cl2_rna_test_clean <- cl2_rna_test[, !names(cl2_rna_test) %in% "ID"]
cl1_img_train_clean <- cl1_img_train[, !names(cl1_img_train) %in% "ID"]
cl1_img_test_clean <- cl1_img_test[, !names(cl1_img_test) %in% "ID"]

cl1_train_clean$status_3y <- factor(cl1_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))
cl2_train_clean$status_3y <- factor(cl2_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))
cl2_rna_train_clean$status_3y <- factor(cl2_rna_train_clean$status_3y, 
                                    levels = c(0, 1),
                                    labels = c("Alive", "Died"))
cl1_img_train_clean$status_3y <- factor(cl1_img_train_clean$status_3y, 
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
char_cols <- sapply(cl1_img_train_clean, is.character)
if (any(char_cols)) {
  cat("Converting character columns to factors:", names(char_cols)[char_cols], "\n")
  cl1_img_train_clean[char_cols] <- lapply(cl1_img_train_clean[char_cols], as.factor)
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

#Train the model - cl1_img
set.seed(123)
cl1_img_xgb_tune <- train(
  status_3y ~ .,
  data = cl1_img_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0)

#Best parameters
print(cl1_img_xgb_tune$bestTune)

#save tuned datasets so no need to run again
#saveRDS(cl1_xgb_tune, file = "Data/Derived/cl1_xgb_tune.rds")
#saveRDS(cl2_xgb_tune, file = "Data/Derived/cl2_xgb_tune.rds")
#saveRDS(cl2_rna_xgb_tune, file = "Data/Derived/cl2_rna_xgb_tune.rds")
#saveRDS(cl1_img_xgb_tune, file = "Data/Derived/cl1_img_xgb_tune.rds")

#read in tuned datasets (new day - not in environment)
cl1_xgb_tune <- readRDS("Data/Derived/cl1_xgb_tune.rds")
cl2_xgb_tune <- readRDS("Data/Derived/cl2_xgb_tune.rds")
cl2_rna_xgb_tune <- readRDS("Data/Derived/cl2_rna_xgb_tune.rds")
cl1_img_xgb_tune <- readRDS("Data/Derived/cl1_img_xgb_tune.rds")



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

cl1_img_xbg_model <- train(
  status_3y ~ .,
  data = cl1_img_train_clean,
  method = "xgbTree",
  trControl = trainControl(
    method = "none", 
    classProbs = TRUE),
  tuneGrid = cl1_img_xgb_tune$bestTune,  
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

#clinical 1 and image data
predictions_prob_cl1_img <- predict(cl1_img_xbg_model, 
                                newdata = cl1_img_test_clean, 
                                type = "prob")[, "Died"]

predictions_class_cl1_img <- predict(cl1_img_xbg_model, newdata = cl1_img_test_clean)

head(data.frame(
  Probability = predictions_prob_cl1_img,
  Predicted_Class = predictions_class_cl1_img))


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
  Model = "A: Clinical")

#For cl2 model
actual_cl2 <- cl2_test_clean$status_3y
roc_cl2 <- roc(actual_cl2, predictions_prob_cl2)

roc_df_cl2 <- data.frame(
  specificity = roc_cl2$specificities,
  sensitivity = roc_cl2$sensitivities,
  Model = "B: Clinical")

#For cl2_rna model
actual_cl2_rna <- cl2_rna_test_clean$status_3y
roc_cl2_rna <- roc(actual_cl2_rna, predictions_prob_cl2_rna)

roc_df_cl2_rna <- data.frame(
  specificity = roc_cl2_rna$specificities,
  sensitivity = roc_cl2_rna$sensitivities,
  Model = "B: Clinical + Genomics")

#For cl1_img model
actual_cl1_img <- cl1_img_test_clean$status_3y 
roc_cl1_img <- roc(actual_cl1_img, predictions_prob_cl1_img)

roc_df_cl1_img <- data.frame(
  specificity = roc_cl1_img$specificities,
  sensitivity = roc_cl1_img$sensitivities,
  Model = "A: Clinical + Imaging")

#combine
roc_df <- rbind(roc_df_cl1, roc_df_cl2, roc_df_cl2_rna, roc_df_cl1_img)

#store AUC values
auc_cl1 <- auc(roc_cl1)
auc_cl2 <- auc(roc_cl2)
auc_cl2_rna <- auc(roc_cl2_rna)
auc_cl1_img <- auc(roc_cl1_img)

print(auc_cl1)
print(auc_cl2)
print(auc_cl2_rna)
print(auc_cl1_img)

roc_df$Model <- factor(
  roc_df$Model,
  levels = c(
    "A: Clinical",
    "B: Clinical",
    "B: Clinical + Genomics",
    "A: Clinical + Imaging"))

png("Output/xgboost_roc_curves.png", width = 8, height = 8, units = "in", res = 300)
plot(1 - roc_df_cl1$specificity, roc_df_cl1$sensitivity, 
     type = "n",
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Figure 1: ROC Curves for XGBoost Models",
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)",
     cex.main = 1.8, cex.lab = 1.5, cex.axis = 1.3)
abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 1.5)
grid(col = "lightgray", lty = 1)
lines(1 - roc_df_cl1$specificity, roc_df_cl1$sensitivity, col = "#2E86AB", lwd = 2)
lines(1 - roc_df_cl2$specificity, roc_df_cl2$sensitivity, col = "#A23B72", lwd = 2)
lines(1 - roc_df_cl2_rna$specificity, roc_df_cl2_rna$sensitivity, col = "red", lwd = 2)
lines(1 - roc_df_cl1_img$specificity, roc_df_cl1_img$sensitivity, col = "blue", lwd = 2)
legend("bottomright", 
       legend = c(paste0("A: Clinical (AUC = ", sprintf("%.3f", auc_cl1), ")"),
                  paste0("B: Clinical (AUC = ", sprintf("%.3f", auc_cl2), ")"),
                  paste0("B: Clinical + Genomics (AUC = ", sprintf("%.3f", auc_cl2_rna), ")"),
                  paste0("A: Clinical + Imaging (AUC = ", sprintf("%.3f", auc_cl1_img), ")")),
       col = c("#2E86AB", "#A23B72", "red", "blue"),
       lwd = 3, cex = 1.1, pt.cex = 1.4, bg = "white", box.col = "gray50", box.lwd = 1.5, inset = c(0.02, 0.02))
dev.off()


#################
#    Brier      #
#################

brier_cl1 <- mean((predictions_prob_cl1 - actual_cl1)^2)

brier_cl2 <- mean((predictions_prob_cl2 - actual_cl2)^2)

brier_cl2_rna <- mean((predictions_prob_cl2_rna - actual_cl2_rna)^2)

brier_cl1_img <- mean((predictions_prob_cl1_img - actual_cl1_img)^2)

cat("Brier Score for Clinical Model 1:", round(brier_cl1, 4), "\n")
cat("Brier Score for Clinical Model 2:", round(brier_cl2, 4), "\n")
cat("Brier Score for Clinical & RNA Model 3:", round(brier_cl2_rna, 4), "\n")
cat("Brier Score for Clinical & Image Model 4:", round(brier_cl1_img, 4), "\n")


#=0.2238, 0.2418, 0.2078, 0.2289 'moderate'

#####################
# Calibration curve #
#####################

df_cl1 <- data.frame(
  pred = predictions_prob_cl1,
  actual = actual_cl1,
  Model = "A: Clinical")

df_cl2 <- data.frame(
  pred = predictions_prob_cl2,
  actual = actual_cl2,
  Model = "B: Clinical")

df_cl2_rna <- data.frame(
  pred = predictions_prob_cl2_rna,
  actual = actual_cl2_rna,
  Model = "B: Clinical + Genomics")

df_cl1_img <- data.frame(
  pred = predictions_prob_cl1_img,
  actual = actual_cl1_img,
  Model = "A: Clinical + Imaging")

combined_df <- rbind(df_cl1, df_cl2, df_cl2_rna,df_cl1_img)

ggplot(combined_df, aes(x = pred, y = actual, color = Model)) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2, size = 1.2, span = 1.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 1) +
  scale_color_manual(values = c("A: Clinical + Imaging" = "#2E86AB", 
                                "B: Clinical" = "#A23B72",
                                "B: Clinical + Genomics" = "red",
                                "A: Clinical + Imaging" = "blue")) +
  labs(x = "Predicted Probability of Death",
       y = "Observed Proportion of Death",
       title = "Calibration Curves with LOESS Smoothing") +
  theme_minimal() +
  theme(legend.position = c(0.2, 0.8)) +
  coord_cartesian(xlim = c(0, 0.8), ylim = c(0, 1))

ggsave("Output/all_models_calib.png", plot = p2)

# Create binned calibration data for all models
create_cal_df <- function(pred, actual, model_name, n_bins = 10) {
  data.frame(pred = pred, actual = actual) %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred),
      observed = mean(actual),
      se = sqrt(observed * (1 - observed) / n()),
      lower = observed - 1.96 * se,
      upper = observed + 1.96 * se,
      Model = model_name,
      .groups = 'drop'
    )
}

cal_combined <- bind_rows(
  create_cal_df(predictions_prob_cl1, actual_cl1, "Clinical Model 1"),
  create_cal_df(predictions_prob_cl2, actual_cl2, "Clinical Model 2"),
  create_cal_df(predictions_prob_cl2_rna, actual_cl2_rna, "Clinical & RNA Model 3"),
  create_cal_df(predictions_prob_cl1_img, actual_cl1_img, "Clinical & Image Model 4")
)




# Create calibration data with 5 bins
create_calibration_data_base <- function(pred, actual, n_bins = 5) {
  breaks <- unique(quantile(pred, probs = seq(0, 1, 1/n_bins)))
  bin_indices <- cut(pred, breaks = breaks, include.lowest = TRUE)
  
  mean_pred <- tapply(pred, bin_indices, mean)
  observed <- tapply(actual, bin_indices, mean)
  n <- tapply(actual, bin_indices, length)
  se <- sqrt(observed * (1 - observed) / n)
  
  return(list(mean_pred = mean_pred, observed = observed, se = se))
}

cal1 <- create_calibration_data_base(predictions_prob_cl1, actual_cl1, n_bins = 5)
cal2 <- create_calibration_data_base(predictions_prob_cl2, actual_cl2, n_bins = 5)
cal2_rna <- create_calibration_data_base(predictions_prob_cl2_rna, actual_cl2_rna, n_bins = 5)
cal1_img <- create_calibration_data_base(predictions_prob_cl1_img, actual_cl1_img, n_bins = 5)

png("Output/calibration_curves_clean.png", width = 10, height = 8, units = "in", res = 300)

plot(0, 0, type = "n", 
     xlim = c(0, 0.8), ylim = c(0, 1),
     main = "Figure 2: Calibration Curves for XGBoost Models",
     xlab = "Predicted Probability of Death",
     ylab = "Observed Proportion of Death",
     cex.main = 1.8, cex.lab = 1.5, cex.axis = 1.3)

abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 2)
grid(col = "lightgray", lty = 1)

lines(cal1$mean_pred, cal1$observed, col = "#2E86AB", lwd = 3)
lines(cal2$mean_pred, cal2$observed, col = "#A23B72", lwd = 3)
lines(cal2_rna$mean_pred, cal2_rna$observed, col = "#D62828", lwd = 3)
lines(cal1_img$mean_pred, cal1_img$observed, col = "#003F88", lwd = 3)

points(cal1$mean_pred, cal1$observed, col = "#2E86AB", pch = 19, cex = 1.5)
points(cal2$mean_pred, cal2$observed, col = "#A23B72", pch = 19, cex = 1.5)
points(cal2_rna$mean_pred, cal2_rna$observed, col = "#D62828", pch = 19, cex = 1.5)
points(cal1_img$mean_pred, cal1_img$observed, col = "#003F88", pch = 19, cex = 1.5)

legend("bottomright",
       legend = c("A: Clinical", "B: Clinical", 
                  "B: Clinical + Genomics", "A: Clinical + Imaging"),
       col = c("#2E86AB", "#A23B72", "#D62828", "#003F88"),
       lwd = 3, pch = 19, pt.cex = 1.2, cex = 1.1, 
       bg = "white", box.col = "gray50")

dev.off()

png("Output/calibration_curves_base.png", width = 10, height = 8, units = "in", res = 300)

plot(0, 0, type = "n", 
     xlim = c(0, 0.8), ylim = c(0, 1),
     main = "Calibration Curves for XGBoost Models",
     xlab = "Predicted Probability of Death",
     ylab = "Observed Proportion of Death",
     cex.main = 1.8, cex.lab = 1.5, cex.axis = 1.3)
abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 2)
grid(col = "lightgray", lty = 1)
add_calibration_curve <- function(cal_data, color, model_name) {
  points(cal_data$mean_pred, cal_data$observed, col = color, pch = 19, cex = 1.5)
  lines(cal_data$mean_pred, cal_data$observed, col = color, lwd = 2.5)
  # Add error bars
  arrows(cal_data$mean_pred, cal_data$observed - 1.96 * cal_data$se,
         cal_data$mean_pred, cal_data$observed + 1.96 * cal_data$se,
         length = 0.05, angle = 90, code = 3, col = color, lwd = 1.5)
}

add_calibration_curve(cal1, "#2E86AB", "A: Clinical")
add_calibration_curve(cal2, "#A23B72", "B: Clinical")
add_calibration_curve(cal2_rna, "#D62828", "B: Clinical + Genomics")
add_calibration_curve(cal1_img, "#003F88", "A: Clinical + Imaging")
legend("bottomright",
       legend = c("A: Clinical", "B: Clinical", 
                  "B: Clinical + Genomics", "A: Clinical + Imaging"),
       col = c("#2E86AB", "#A23B72", "#D62828", "#003F88"),
       pch = 19, lwd = 2.5, cex = 1.1, bg = "white", box.col = "gray50")

dev.off()


#loess smoothing instead of binning
png("Output/xgboost_cal_curves_smooth.png", width = 10, height = 8, units = "in", res = 300)

plot(0, 0, type = "n", 
     xlim = c(0, 0.8), ylim = c(0, 1),
     main = "Figure 2: Calibration Curves for XGBoost Models",
     xlab = "Predicted Probability of Death",
     ylab = "Observed Proportion of Death",
     cex.main = 1.8, cex.lab = 1.5, cex.axis = 1.3)

abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 2)
grid(col = "lightgray", lty = 1)

add_smooth_curve <- function(pred, actual, color) {
  loess_fit <- loess(actual ~ pred, degree = 1, span = 1.2)
  pred_sorted <- sort(pred)
  fit_smooth <- predict(loess_fit, newdata = data.frame(pred = pred_sorted), se = TRUE)
  
  lines(pred_sorted, fit_smooth$fit, col = color, lwd = 3)
  
  polygon(c(pred_sorted, rev(pred_sorted)), 
          c(fit_smooth$fit + 1.96 * fit_smooth$se, 
            rev(fit_smooth$fit - 1.96 * fit_smooth$se)),
          col = adjustcolor(color, alpha.f = 0.15), border = NA)
}

add_smooth_curve(predictions_prob_cl1, actual_cl1, "#2E86AB")
add_smooth_curve(predictions_prob_cl2, actual_cl2, "#A23B72")
add_smooth_curve(predictions_prob_cl2_rna, actual_cl2_rna, "#D62828")
add_smooth_curve(predictions_prob_cl1_img, actual_cl1_img, "#003F88")

legend("bottomright",
       legend = c("A: Clinical", "B: Clinical", 
                  "B: Clinical + Genomics", "A: Clinical + Imaging"),
       col = c("#2E86AB", "#A23B72", "#D62828", "#003F88"),
       lwd = 3, cex = 1.1, bg = "white", box.col = "gray50")

dev.off()

calculate_calibration_metrics <- function(pred, actual) {
  cal_model <- lm(actual ~ pred)
  slope <- coef(cal_model)[2]
  intercept <- coef(cal_model)[1]
  r_squared <- summary(cal_model)$r.squared
  return(c(slope = slope, intercept = intercept, r_squared = r_squared))
}

metrics_cl1 <- calculate_calibration_metrics(predictions_prob_cl1, actual_cl1)
metrics_cl2 <- calculate_calibration_metrics(predictions_prob_cl2, actual_cl2)
metrics_cl2_rna <- calculate_calibration_metrics(predictions_prob_cl2_rna, actual_cl2_rna)
metrics_cl1_img <- calculate_calibration_metrics(predictions_prob_cl1_img, actual_cl1_img)

# Print metrics for report
print(metrics_cl1)
print(metrics_cl2)
print(metrics_cl2_rna)
print(metrics_cl1_img)


##### perfomance metrics

#Function to calculate metrics at optimal threshold
calculate_metrics_at_optimal_threshold <- function(actual, predicted_prob, model_name) {
  
  roc_obj <- roc(actual, predicted_prob)
  optimal_coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
  optimal_threshold <- optimal_coords$threshold
  
  actual_factor <- factor(actual, levels = c(0, 1), labels = c("Alive", "Died"))
  predicted_class <- factor(ifelse(predicted_prob > optimal_threshold, "Died", "Alive"),
                            levels = c("Alive", "Died"))
  
  # Calculate confusion matrix
  cm <- confusionMatrix(predicted_class, actual_factor, positive = "Died")
  
  data.frame(
    Model = model_name,
    Optimal_Threshold = round(optimal_threshold, 3),
    AUC = round(auc(roc_obj), 3),
    Accuracy = round(cm$overall["Accuracy"], 3),
    Sensitivity = round(cm$byClass["Sensitivity"], 3),
    Specificity = round(cm$byClass["Specificity"], 3),
    Precision = round(cm$byClass["Precision"], 3),
    NPV = round(cm$byClass["Neg Pred Value"], 3),
    F1 = round(cm$byClass["F1"], 3)
  )
}

metrics_optimal_cl1 <- calculate_metrics_at_optimal_threshold(actual_cl1, predictions_prob_cl1, "A: Clinical")
metrics_optimal_cl2 <- calculate_metrics_at_optimal_threshold(actual_cl2, predictions_prob_cl2, "B: Clinical")
metrics_optimal_cl2_rna <- calculate_metrics_at_optimal_threshold(actual_cl2_rna, predictions_prob_cl2_rna, "B: Clinical + Genomics")
metrics_optimal_cl1_img <- calculate_metrics_at_optimal_threshold(actual_cl1_img, predictions_prob_cl1_img, "A: Clinical + Imaging")

all_metrics_optimal <- rbind(metrics_optimal_cl1, metrics_optimal_cl2, 
                             metrics_optimal_cl2_rna, metrics_optimal_cl1_img)

comparison_table <- data.frame(
  Model = all_metrics_optimal$Model,
  AUC = all_metrics_optimal$AUC,
  Optimal_Threshold = all_metrics_optimal$Optimal_Threshold,
  `Sens@0.5` = c(1.000, 0.222, 0.000, 0.873),
  `Spec@0.5` = c(0.000, 0.826, 1.000, 0.100),
  `Sens@Optimal` = all_metrics_optimal$Sensitivity,
  `Spec@Optimal` = all_metrics_optimal$Specificity,
  `F1@Optimal` = all_metrics_optimal$F1
)

print(comparison_table)


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

#calculate class imbalance ratio for cl1_img
cl1_img_class_counts <- table(cl1_img_train_clean$status_3y)
cl1_img_scale_pos_weight <- cl1_img_class_counts["Alive"] / cl1_img_class_counts["Died"]
cat("cl1_img - Alive:", cl1_img_class_counts["Alive"], 
    "Died:", cl1_img_class_counts["Died"], 
    "scale_pos_weight:", cl1_img_scale_pos_weight, "\n")

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

#train cl1_img with scale_pos_weight
set.seed(123)
cl1_img_xgb_tune_weighted <- train(
  status_3y ~ .,
  data = cl1_img_train_clean,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid_weighted,
  metric = "ROC",
  na.action = na.pass,
  nthread = 4,
  verbose = 1,
  verbosity = 0,
  scale_pos_weight = cl1_img_scale_pos_weight)

#save tuned models so don't need to run again
#saveRDS(cl1_xgb_tune_weighted, file = "Data/Derived/cl1_xgb_tune_weighted.rds")
#saveRDS(cl2_xgb_tune_weighted, file = "Data/Derived/cl2_xgb_tune_weighted.rds")
#saveRDS(cl2_rna_xgb_tune_weighted, file = "Data/Derived/cl2_rna_xgb_tune_weighted.rds")
#saveRDS(cl1_img_xgb_tune_weighted, file = "Data/Derived/cl1_img_xgb_tune_weighted.rds")

#read in tuned datasets (new day - not in environment)
cl1_xgb_tune_weighted <- readRDS("Data/Derived/cl1_xgb_tune_weighted.rds")
cl2_xgb_tune_weighted <- readRDS("Data/Derived/cl2_xgb_tune_weighted.rds")
cl2_rna_xgb_tune_weighted <- readRDS("Data/Derived/cl2_rna_xgb_tune_weighted.rds")
cl1_img_xgb_tune_weighted <- readRDS("Data/Derived/cl1_img_xgb_tune_weighted.rds")


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

cl1_img_xgb_best_weighted <- train(
  status_3y ~ .,
  data = cl1_img_train_clean,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl1_img_xgb_tune_weighted$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0,
  scale_pos_weight = cl1_img_scale_pos_weight)

#compare performance
cat("Without weighting - ROC:", max(cl1_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl1_xgb_tune_weighted$results$ROC), "\n")

cat("Without weighting - ROC:", max(cl2_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl2_xgb_tune_weighted$results$ROC), "\n")

cat("Without weighting - ROC:", max(cl2_rna_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl2_rna_xgb_tune_weighted$results$ROC), "\n")

cat("Without weighting - ROC:", max(cl1_img_xgb_tune$results$ROC), "\n")
cat("With weighting - ROC:", max(cl1_img_xgb_tune_weighted$results$ROC), "\n")

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

pred_cl1_img_original <- predict(cl1_img_xbg_model, newdata = cl1_img_test_clean, type = "prob")[, "Died"]
pred_cl1_img_weighted <- predict(cl1_img_xgb_best_weighted, newdata = cl1_img_test_clean, type = "prob")[, "Died"]

roc_cl1_original <- roc(cl1_test_clean$status_3y, pred_cl1_original)
roc_cl1_weighted <- roc(cl1_test_clean$status_3y, pred_cl1_weighted)
roc_cl2_original <- roc(cl2_test_clean$status_3y, pred_cl2_original)
roc_cl2_weighted <- roc(cl2_test_clean$status_3y, pred_cl2_weighted)
roc_cl2_rna_original <- roc(cl2_rna_test_clean$status_3y, pred_cl2_rna_original)
roc_cl2_rna_weighted <- roc(cl2_rna_test_clean$status_3y, pred_cl2_rna_weighted)
roc_cl1_img_original <- roc(cl1_img_test_clean$status_3y, pred_cl1_img_original)
roc_cl1_img_weighted <- roc(cl1_img_test_clean$status_3y, pred_cl1_img_weighted)


test_results <- data.frame(
  Model = c("A: Clinical", "A: Clinical", 
            "B: Clinical", "B: Clinical",
            "B: Clinical + Genomics", "B: Clinical + Genomics",
            "A: Clinical + Imaging", "A: Clinical + Imaging"),
  Version = rep(c("Unweighted", "Weighted"), 4),
  Test_AUC = c(
    auc(roc_cl1_original),
    auc(roc_cl1_weighted),
    auc(roc_cl2_original),
    auc(roc_cl2_weighted),
    auc(roc_cl2_rna_original),
    auc(roc_cl2_rna_weighted),
    auc(roc_cl1_img_original),
    auc(roc_cl1_img_weighted)
  ),
  CI_lower = c(
    ci.auc(roc_cl1_original)[1],
    ci.auc(roc_cl1_weighted)[1],
    ci.auc(roc_cl2_original)[1],
    ci.auc(roc_cl2_weighted)[1],
    ci.auc(roc_cl2_rna_original)[1],
    ci.auc(roc_cl2_rna_weighted)[1],
    ci.auc(roc_cl1_img_original)[1],
    ci.auc(roc_cl1_img_weighted)[1]
  ),
  CI_upper = c(
    ci.auc(roc_cl1_original)[3],
    ci.auc(roc_cl1_weighted)[3],
    ci.auc(roc_cl2_original)[3],
    ci.auc(roc_cl2_weighted)[3],
    ci.auc(roc_cl2_rna_original)[3],
    ci.auc(roc_cl2_rna_weighted)[3],
    ci.auc(roc_cl1_img_original)[3],
    ci.auc(roc_cl1_img_weighted)[3]
  )
)

test_results$Test_AUC <- round(test_results$Test_AUC, 3)
test_results$CI_lower <- round(test_results$CI_lower, 3)
test_results$CI_upper <- round(test_results$CI_upper, 3)

test_results$AUC_Report <- sprintf("%.3f (95%% CI: %.3f-%.3f)", 
                                   test_results$Test_AUC, 
                                   test_results$CI_lower, 
                                   test_results$CI_upper)

print(test_results[, c("Model", "Version", "AUC_Report")])



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

#Upsample cl2_rna
set.seed(123)
cl2_rna_train_upsampled <- upSample(
  x = cl2_rna_train_clean[, !names(cl2_rna_train_clean) %in% "status_3y"],
  y = cl2_rna_train_clean$status_3y,
  yname = "status_3y")

#Upsample cl1_img
set.seed(123)
cl1_img_train_upsampled <- upSample(
  x = cl1_img_train_clean[, !names(cl1_img_train_clean) %in% "status_3y"],
  y = cl1_img_train_clean$status_3y,
  yname = "status_3y")

# Check balance
table(cl1_train_upsampled$status_3y)
table(cl2_train_upsampled$status_3y)
table(cl2_rna_train_upsampled$status_3y)

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

set.seed(123)
cl2_rna_xgb_upsampled <- train(
  status_3y ~ .,
  data = cl2_rna_train_upsampled,
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

set.seed(123)
cl1_img_xgb_upsampled <- train(
  status_3y ~ .,
  data = cl1_img_train_upsampled,
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

#save tuned models so don't need to run again
#saveRDS(cl1_xgb_upsampled, file = "Data/Derived/cl1_xgb_tune_upsampled.rds")
#saveRDS(cl2_xgb_upsampled, file = "Data/Derived/cl2_xgb_tune_upsampled.rds")
#saveRDS(cl2_rna_xgb_upsampled, file = "Data/Derived/cl2_rna_xgb_tune_upsampled.rds")
#saveRDS(cl1_img_xgb_upsampled, file = "Data/Derived/cl1_img_xgb_tune_upsampled.rds")

#read in tuned datasets (new day - not in environment)
cl1_xgb_upsampled <- readRDS("Data/Derived/cl1_xgb_tune_upsampled.rds")
cl2_xgb_upsampled <- readRDS("Data/Derived/cl2_xgb_tune_upsampled.rds")
cl2_rna_xgb_upsampled <- readRDS("Data/Derived/cl2_rna_xgb_tune_upsampled.rds")
cl1_img_xgb_upsampled <- readRDS("Data/Derived/cl1_img_xgb_tune_upsampled.rds")

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

cl2_rna_best_upsampled <- train(
  status_3y ~ .,
  data = cl2_rna_train_upsampled,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl2_rna_xgb_upsampled$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0)

cl1_img_best_upsampled <- train(
  status_3y ~ .,
  data = cl1_img_train_upsampled,
  method = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = cl1_img_xgb_upsampled$bestTune,
  na.action = na.pass,
  nthread = 4,
  verbose = 0)

pred_cl1_original <- predict(cl1_xbg_model, newdata = cl1_test_clean, type = "prob")[, "Died"]
pred_cl1_upsampled <- predict(cl1_best_upsampled, newdata = cl1_test_clean, type = "prob")[, "Died"]

pred_cl2_original <- predict(cl2_xbg_model, newdata = cl2_test_clean, type = "prob")[, "Died"]
pred_cl2_upsampled <- predict(cl2_best_upsampled, newdata = cl2_test_clean, type = "prob")[, "Died"]

pred_cl2_rna_original <- predict(cl2_rna_xbg_model, newdata = cl2_rna_test_clean, type = "prob")[, "Died"]
pred_cl2_rna_upsampled <- predict(cl2_rna_best_upsampled, newdata = cl2_rna_test_clean, type = "prob")[, "Died"]

pred_cl1_img_original <- predict(cl1_img_xbg_model, newdata = cl1_img_test_clean, type = "prob")[, "Died"]
pred_cl1_img_upsampled <- predict(cl1_img_best_upsampled, newdata = cl1_img_test_clean, type = "prob")[, "Died"]


roc_cl1_original <- roc(cl1_test_clean$status_3y, pred_cl1_original)
roc_cl1_upsampled <- roc(cl1_test_clean$status_3y, pred_cl1_upsampled)

roc_cl2_original <- roc(cl2_test_clean$status_3y, pred_cl2_original)
roc_cl2_upsampled <- roc(cl2_test_clean$status_3y, pred_cl2_upsampled)

roc_cl2_rna_original <- roc(cl2_rna_test_clean$status_3y, pred_cl2_rna_original)
roc_cl2_rna_upsampled <- roc(cl2_rna_test_clean$status_3y, pred_cl2_rna_upsampled)

roc_cl1_img_original <- roc(cl1_img_test_clean$status_3y, pred_cl1_img_original)
roc_cl1_img_upsampled <- roc(cl1_img_test_clean$status_3y, pred_cl1_img_upsampled)


cat("CL1 Original AUC:", round(auc(roc_cl1_original), 4), "\n")
cat("CL1 Upsampled AUC:", round(auc(roc_cl1_upsampled), 4), "\n")
cat("CL2 Original AUC:", round(auc(roc_cl2_original), 4), "\n")
cat("CL2 Upsampled AUC:", round(auc(roc_cl2_upsampled), 4), "\n")
cat("CL2_rna Original AUC:", round(auc(roc_cl2_rna_original), 4), "\n")
cat("CL2_rna Upsampled AUC:", round(auc(roc_cl2_rna_upsampled), 4), "\n")
cat("CL1_img Original AUC:", round(auc(roc_cl1_img_original), 4), "\n")
cat("CL1_img Upsampled AUC:", round(auc(roc_cl1_img_upsampled), 4), "\n")

auc_cl1_original <- auc(roc_cl1_original)
ci_cl1_original <- ci.auc(roc_cl1_original)
auc_cl2_original <- auc(roc_cl2_original)
ci_cl2_original <- ci.auc(roc_cl2_original)
auc_cl2_rna_original <- auc(roc_cl2_rna_original)
ci_cl2_rna_original <- ci.auc(roc_cl2_rna_original)
auc_cl1_img_original <- auc(roc_cl1_img_original)
ci_cl1_img_original <- ci.auc(roc_cl1_img_original)
auc_cl1_upsampled <- auc(roc_cl1_upsampled)
ci_cl1_upsampled <- ci.auc(roc_cl1_upsampled)
auc_cl2_upsampled <- auc(roc_cl2_upsampled)
ci_cl2_upsampled <- ci.auc(roc_cl2_upsampled)
auc_cl2_rna_upsampled <- auc(roc_cl2_rna_upsampled)
ci_cl2_rna_upsampled <- ci.auc(roc_cl2_rna_upsampled)
auc_cl1_img_upsampled <- auc(roc_cl1_img_upsampled)
ci_cl1_img_upsampled <- ci.auc(roc_cl1_img_upsampled)

test_results <- data.frame(
  Model = c(
    "A: Clinical", "A: Clinical",
    "B: Clinical", "B: Clinical",
    "B: Clinical + Genomics", "B: Clinical + Genomics",
    "A: Clinical + Imaging", "A: Clinical + Imaging"
  ),
  Version = rep(c("Original", "Upsampled"), 4),
  Test_AUC = c(
    auc_cl1_original, auc_cl1_upsampled,
    auc_cl2_original, auc_cl2_upsampled,
    auc_cl2_rna_original, auc_cl2_rna_upsampled,
    auc_cl1_img_original, auc_cl1_img_upsampled
  ),
  CI_lower = c(
    ci_cl1_original[1], ci_cl1_upsampled[1],
    ci_cl2_original[1], ci_cl2_upsampled[1],
    ci_cl2_rna_original[1], ci_cl2_rna_upsampled[1],
    ci_cl1_img_original[1], ci_cl1_img_upsampled[1]
  ),
  CI_upper = c(
    ci_cl1_original[3], ci_cl1_upsampled[3],
    ci_cl2_original[3], ci_cl2_upsampled[3],
    ci_cl2_rna_original[3], ci_cl2_rna_upsampled[3],
    ci_cl1_img_original[3], ci_cl1_img_upsampled[3]
  )
)

test_results$Test_AUC <- round(test_results$Test_AUC, 3)
test_results$CI_lower <- round(test_results$CI_lower, 3)
test_results$CI_upper <- round(test_results$CI_upper, 3)

test_results$AUC_Report <- sprintf("%.3f (95%% CI: %.3f-%.3f)", 
                                   test_results$Test_AUC, 
                                   test_results$CI_lower, 
                                   test_results$CI_upper)

print(test_results[, c("Model", "Version", "AUC_Report")])


cat("With upsampling - ROC:", max(cl1_best_upsampled$results$ROC), "\n")
cat("With upsampling - ROC:", max(cl2_best_upsampled$results$ROC), "\n")
cat("With upsampling - ROC:", max(cl2_rna_best_upsampled$results$ROC), "\n")
cat("With upsampling - ROC:", max(cl1_img_best_upsampled$results$ROC), "\n")


