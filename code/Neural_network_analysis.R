#############################################################################
#                                                                           #
# Program Name:  Neural Network Analysis                                    #
#
#  Outputs:      mlp_roc_curves.png, mlp_cal_curves_smooth.png, pfi_importance_top5_minimal_white.png
#                                                                           #
#############################################################################


###############################
##      LOAD PACKAGES       ##
##############################
library(ggplot2)
library(glmnet)
library(pROC)
library(caret)
library(dplyr)
library(tidyr)


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


#remove site_num for cl2 datasets - no predictive ability as all VA
cl2_train <- cl2_train[, !(names(cl2_train) %in% c("alk_num","site_num"))]
cl2_test <- cl2_test[, !(names(cl2_test) %in% c("alk_num","site_num"))]
cl2_rna_train <- cl2_rna_train[, !(names(cl2_rna_train) %in% c("alk_num","site_num"))]
cl2_rna_test <- cl2_rna_test[, !(names(cl2_rna_test) %in% c("alk_num","site_num"))]


set.seed(123)

#prepare data with dummy variables
prepare_data <- function(data, dummy_model = NULL, train = TRUE) {
  data_clean <- data[, !names(data) %in% c("ID", "X")]
  
  if(all(c("T_stage", "N_stage", "M_stage", "overall_stage") %in% names(data_clean))) {
    #---- cl1 structure ----
    numeric_vars <- data_clean[, c("T_stage", "N_stage", "M_stage", "overall_stage")]
    categorical_vars <- c("gender_num", "hist_num")
  } else {
    #---- cl2 structure ----
    numeric_vars <- data_clean[, c(
      "Age", "tl_RUL", "tl_RML", "tl_RLL",
      "tl_LUL", "tl_LLL", "tl_LLing", "tl_U",
      "pleural", "adj_trt", "Chemotherapy",
      "Radiation", "Recurrence", "weight",
      "pack_yrs", "quit_smoke_yr")]
    categorical_vars <- c(
      "gender_num", "hist_num",
      "ethnic_num", "XGG_num", "path_t_num",
      "path_m_num", "path_n_num", "lymph_num",
      "egfr_num", "kras_num",
      "smoke_num")}
  dummy_formula <- as.formula(
    paste("~", paste(categorical_vars, collapse = " + ")))
  
  if(train) {
    dummy_model <- dummyVars(dummy_formula,
                             data = data_clean,
                             sep = "_")
    dummy_vars <- predict(dummy_model, newdata = data_clean)
    return(list(
      x = cbind(numeric_vars, dummy_vars),
      y = data_clean$status_3y,
      dummy_model = dummy_model
    ))
  } else {
    dummy_vars <- predict(dummy_model, newdata = data_clean)
    return(list(
      x = cbind(numeric_vars, dummy_vars),
      y = data_clean$status_3y))
  }
}

#prepare training data
cl1_train_prep <- prepare_data(cl1_train, train = TRUE)
cl1_x_train <- as.matrix(cl1_train_prep$x)
cl1_y_train <- cl1_train_prep$y

cl2_train_prep <- prepare_data(cl2_train, train = TRUE)
cl2_x_train <- as.matrix(cl2_train_prep$x)
cl2_y_train <- cl2_train_prep$y

cl2_rna_train_prep <- prepare_data(cl2_rna_train, train = TRUE)
cl2_rna_x_train <- as.matrix(cl2_rna_train_prep$x)
cl2_rna_y_train <- cl2_rna_train_prep$y

cl1_img_train_prep <- prepare_data(cl1_img_train, train = TRUE)
cl1_img_x_train <- as.matrix(cl1_img_train_prep$x)
cl1_img_y_train <- cl1_img_train_prep$y

#prepare test data
cl1_test_prep <- prepare_data(cl1_test, dummy_model = cl1_train_prep$dummy_model, train = FALSE)
cl1_x_test <- as.matrix(cl1_test_prep$x)
cl1_y_test <- cl1_test_prep$y

cl2_test_prep <- prepare_data(cl2_test, dummy_model = cl2_train_prep$dummy_model, train = FALSE)
cl2_x_test <- as.matrix(cl2_test_prep$x)
cl2_y_test <- cl2_test_prep$y

cl2_rna_test_prep <- prepare_data(cl2_rna_test, dummy_model = cl2_rna_train_prep$dummy_model, train = FALSE)
cl2_rna_x_test <- as.matrix(cl2_rna_test_prep$x)
cl2_rna_y_test <- cl2_rna_test_prep$y

cl1_img_test_prep <- prepare_data(cl1_img_test, dummy_model = cl1_img_train_prep$dummy_model, train = FALSE)
cl1_img_x_test <- as.matrix(cl1_img_test_prep$x)
cl1_img_y_test <- cl1_img_test_prep$y


#FUNCTION FOR LASSO FEATURE SELECTION WITH 5-FOLD CV

lasso_feature_selection <- function(x_train, y_train, dataset_name) {
  
  cat("\nProcessing:", dataset_name, "\n")
  
  #lasso CV
  cv_lasso <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = 1,
    type.measure = "auc",
    nfolds = 5
  )
  
  #features with non-zero coefficients at lambda.1se
  coef_matrix <- as.matrix(coef(cv_lasso, s = cv_lasso$lambda.1se))
  selected_features <- rownames(coef_matrix)[coef_matrix[, 1] != 0]
  selected_features <- selected_features[selected_features != "(Intercept)"]
  
  #coefficients
  feature_coefs <- coef_matrix[selected_features, 1]
  
  cat("  Features:", length(selected_features), "/", ncol(x_train), 
      "| Max AUC:", round(max(cv_lasso$cvm), 4), "\n")
  
  if(length(selected_features) > 0) {
    top_idx <- order(abs(feature_coefs), decreasing = TRUE)[1:min(5, length(selected_features))]
    cat("  Top features:\n")
    for(i in top_idx) {
      cat("   ", selected_features[i], ":", round(feature_coefs[i], 4), "\n")
    }
  }
  
  return(list(
    selected_features = selected_features,
    feature_coefficients = feature_coefs,
    n_selected = length(selected_features),
    max_cv_auc = max(cv_lasso$cvm)
  ))
}


set.seed(123)

cl1_lasso <- lasso_feature_selection(cl1_x_train, cl1_y_train, "CL1")

cl2_lasso <- lasso_feature_selection(cl2_x_train, cl2_y_train, "CL2")

cl2_rna_lasso <- lasso_feature_selection(cl2_rna_x_train, cl2_rna_y_train, "CL2_RNA")

cl1_img_lasso <- lasso_feature_selection(cl1_img_x_train, cl1_img_y_train, "CL1_IMG")

summary_table <- data.frame(
  Dataset = c("CL1", "CL2", "CL2_RNA", "CL1_IMG"),
  Total_Features = c(ncol(cl1_x_train), ncol(cl2_x_train), 
                     ncol(cl2_rna_x_train), ncol(cl1_img_x_train)),
  Selected = c(cl1_lasso$n_selected, cl2_lasso$n_selected,
               cl2_rna_lasso$n_selected, cl1_img_lasso$n_selected),
  Max_AUC = c(cl1_lasso$max_cv_auc, cl2_lasso$max_cv_auc,
              cl2_rna_lasso$max_cv_auc, cl1_img_lasso$max_cv_auc))

print(summary_table)


#CREATE REDUCED DATASETS WITH FEATURES OF INTEREST

create_reduced_dataset <- function(x_train, x_test, lasso_result, dataset_name) {
  if(length(lasso_result$selected_features) == 0) {
    cat("\nWARNING: No features selected for", dataset_name, "- using all features\n")
    return(list(
      x_train_reduced = x_train,
      x_test_reduced = x_test,
      selected_features = colnames(x_train)
    ))
  }
  
  train_features <- colnames(x_train)
  test_features <- colnames(x_test)
  
  available_features <- lasso_result$selected_features[lasso_result$selected_features %in% train_features]
  available_features <- available_features[available_features %in% test_features]
  
  if(length(available_features) < length(lasso_result$selected_features)) {
    cat("\nWarning for", dataset_name, ":", 
        length(lasso_result$selected_features) - length(available_features), 
        "features missing from test set\n")
  }
  
  x_train_reduced <- x_train[, available_features, drop = FALSE]
  x_test_reduced <- x_test[, available_features, drop = FALSE]
  
  cat("\n", dataset_name, "reduced dataset:\n")
  cat("  Training: from", ncol(x_train), "to", ncol(x_train_reduced), "features\n")
  cat("  Testing: from", ncol(x_test), "to", ncol(x_test_reduced), "features\n")
  
  return(list(
    x_train_reduced = x_train_reduced,
    x_test_reduced = x_test_reduced,
    selected_features = available_features
  ))
}

cl1_reduced <- create_reduced_dataset(cl1_x_train, cl1_x_test, cl1_lasso, "CL1")
cl2_reduced <- create_reduced_dataset(cl2_x_train, cl2_x_test, cl2_lasso, "CL2")
cl2_rna_reduced <- create_reduced_dataset(cl2_rna_x_train, cl2_rna_x_test, cl2_rna_lasso, "CL2_RNA")
cl1_img_reduced <- create_reduced_dataset(cl1_img_x_train, cl1_img_x_test, cl1_img_lasso, "CL1_IMG")


#HYPERPARAMETER TUNING FUNCTION FOR MLP
library(nnet)

set.seed(123)

tune_mlp_r <- function(x_train, y_train, n_folds = 4, n_trials = 20) {
  
  set.seed(123)
  
  x_train <- as.matrix(x_train)
  
  folds <- createFolds(y_train, k = n_folds, list = TRUE)
  
  best_auc <- 0
  best_params <- NULL
  results <- data.frame()
  
  for(i in 1:n_trials) {
    
    # Random search space (adjusted for your small dataset)
    params <- list(
      size = sample(c(1, 2, 3, 4, 5, 6), 1), 
      decay = 10^runif(1, -4, -1),             
      maxit = sample(c(200, 300, 400, 500), 1)
    )
    
    fold_aucs <- c()
    
    #ross validation
    for(fold in 1:n_folds) {
      
      val_idx <- folds[[fold]]
      train_idx <- setdiff(seq_len(nrow(x_train)), val_idx)
      
      scale_params <- preProcess(x_train[train_idx, ], method = c("center", "scale"))
      x_train_fold <- predict(scale_params, x_train[train_idx, ])
      x_val_fold <- predict(scale_params, x_train[val_idx, ])
      
      model <- nnet(
        x = as.matrix(x_train_fold),
        y = y_train[train_idx],
        size = params$size,
        decay = params$decay,
        maxit = params$maxit,
        trace = FALSE,
        linout = FALSE 
      )
      
      #predict on validation fold
      pred <- predict(model, as.matrix(x_val_fold), type = "raw")
      
      #AUC
      auc_val <- roc(y_train[val_idx], as.vector(pred), quiet = TRUE)$auc
      fold_aucs <- c(fold_aucs, auc_val)
    }
    
    mean_auc <- mean(fold_aucs, na.rm = TRUE)
    
    results <- rbind(results, data.frame(
      trial = i,
      size = params$size,
      decay = round(params$decay, 5),
      maxit = params$maxit,
      mean_auc = round(mean_auc, 4)
    ))
    
    if(mean_auc > best_auc) {
      best_auc <- mean_auc
      best_params <- params
      
      cat(sprintf("Trial %d/%d -> NEW BEST CV AUC: %.4f (size=%d, decay=%.5f)\n", 
                  i, n_trials, mean_auc, params$size, params$decay))
    } else if(i %% 5 == 0) {
      cat(sprintf("Trial %d/%d -> Best so far: %.4f\n", i, n_trials, best_auc))
    }
  }
  
  scale_params <- preProcess(x_train, method = c("center", "scale"))
  x_train_scaled <- as.matrix(predict(scale_params, x_train))
  
  final_model <- nnet(
    x = x_train_scaled,
    y = y_train,
    size = best_params$size,
    decay = best_params$decay,
    maxit = best_params$maxit,
    trace = FALSE,
    linout = FALSE
  )
  
  return(list(
    best_params = best_params,
    best_cv_auc = best_auc,
    tuning_results = results,
    final_model = final_model,
    scale_params = scale_params
  ))
}

cl1_tuning <- tune_mlp_r(cl1_reduced$x_train_reduced,cl1_y_train,n_folds = 4,n_trials = 10)
cl2_tuning <- tune_mlp_r(cl2_reduced$x_train_reduced,cl2_y_train,n_folds = 4,n_trials = 10)
cl2_rna_tuning <- tune_mlp_r(cl2_rna_reduced$x_train_reduced,cl2_rna_y_train,n_folds = 4,n_trials = 10)
cl1_img_tuning <- tune_mlp_r(cl1_img_reduced$x_train_reduced,cl1_img_y_train,n_folds = 4,n_trials = 10)

summary_table <- data.frame(
  Dataset = c("A: Clinical", "B: Clinical", "B: Clinical + Genomics", "A: Clinical + Imaging"),
  Hidden_Neurons = c(cl1_tuning$best_params$size,
                     cl2_tuning$best_params$size,
                     cl2_rna_tuning$best_params$size,
                     cl1_img_tuning$best_params$size),
  Weight_Decay = c(round(cl1_tuning$best_params$decay, 6),
                   round(cl2_tuning$best_params$decay, 6),
                   round(cl2_rna_tuning$best_params$decay, 6),
                   round(cl1_img_tuning$best_params$decay, 6)),
  Max_Iterations = c(cl1_tuning$best_params$maxit,
                     cl2_tuning$best_params$maxit,
                     cl2_rna_tuning$best_params$maxit,
                     cl1_img_tuning$best_params$maxit),
  Best_CV_AUC = c(round(cl1_tuning$best_cv_auc, 4),
                  round(cl2_tuning$best_cv_auc, 4),
                  round(cl2_rna_tuning$best_cv_auc, 4),
                  round(cl1_img_tuning$best_cv_auc, 4))
)

print(summary_table)


#EVALUATE MODEL ON THE TEST SET USING MLP

evaluate_final_nnet <- function(tuning_result, x_test, y_test, dataset_name) {
  
  #Scale test set using training parameters
  x_test_scaled <- as.matrix(predict(tuning_result$scale_params, x_test))
  
  predictions <- as.numeric(predict(tuning_result$final_model, x_test_scaled, type = "raw"))
  
  #AUC-ROC
  roc_obj <- roc(y_test, predictions, quiet = TRUE)
  test_auc <- auc(roc_obj)
  
  #Brier score
  brier_score <- mean((predictions - y_test)^2)
  
  #Optimal threshold and classification metrics
  opt_thresh <- as.numeric(coords(roc_obj, "best", ret = "threshold", transpose = FALSE)[1])
  pred_class <- ifelse(predictions > opt_thresh, 1, 0)
  
  accuracy <- mean(pred_class == y_test)
  
  tp <- sum(pred_class == 1 & y_test == 1)
  tn <- sum(pred_class == 0 & y_test == 0)
  fp <- sum(pred_class == 1 & y_test == 0)
  fn <- sum(pred_class == 0 & y_test == 1)
  
  sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), NA)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), NA)
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), NA)
  
  cat("AUC-ROC:", round(test_auc, 4), "\n")
  cat("Brier Score:", round(brier_score, 4), "\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
  cat("Sensitivity:", round(sensitivity, 4), "\n")
  cat("Specificity:", round(specificity, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("Optimal threshold:", round(opt_thresh, 4), "\n")
  

  return(list(
    dataset = dataset_name,
    test_auc = test_auc,
    brier_score = brier_score,
    predictions = predictions,
    predicted_class = pred_class,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    optimal_threshold = opt_thresh,
    roc_obj = roc_obj,
    y_test = y_test
  ))
}

cl1_eval <- evaluate_final_nnet(cl1_tuning, 
                                cl1_reduced$x_test_reduced, 
                                cl1_y_test, 
                                "CL1")

cl2_eval <- evaluate_final_nnet(cl2_tuning, 
                                cl2_reduced$x_test_reduced, 
                                cl2_y_test, 
                                "CL2")

cl2_rna_eval <- evaluate_final_nnet(cl2_rna_tuning, 
                                    cl2_rna_reduced$x_test_reduced, 
                                    cl2_rna_y_test, 
                                    "CL2_RNA")

cl1_img_eval <- evaluate_final_nnet(cl1_img_tuning, 
                                    cl1_img_reduced$x_test_reduced, 
                                    cl1_img_y_test, 
                                    "CL1_IMG")

comparison_results <- data.frame(
  Dataset = c("CL1", "CL2", "CL2_RNA", "CL1_IMG"),
  Features_After_Lasso = c(ncol(cl1_reduced$x_train_reduced),
                           ncol(cl2_reduced$x_train_reduced),
                           ncol(cl2_rna_reduced$x_train_reduced),
                           ncol(cl1_img_reduced$x_train_reduced)),
  Best_CV_AUC = c(cl1_tuning$best_cv_auc, 
                  cl2_tuning$best_cv_auc,
                  cl2_rna_tuning$best_cv_auc, 
                  cl1_img_tuning$best_cv_auc),
  Test_AUC = c(cl1_eval$test_auc, 
               cl2_eval$test_auc, 
               cl2_rna_eval$test_auc, 
               cl1_img_eval$test_auc),
  Test_Accuracy = c(cl1_eval$accuracy, 
                    cl2_eval$accuracy, 
                    cl2_rna_eval$accuracy, 
                    cl1_img_eval$accuracy),
  Test_Sensitivity = c(cl1_eval$sensitivity, 
                       cl2_eval$sensitivity, 
                       cl2_rna_eval$sensitivity, 
                       cl1_img_eval$sensitivity),
  Test_Specificity = c(cl1_eval$specificity, 
                       cl2_eval$specificity, 
                       cl2_rna_eval$specificity, 
                       cl1_img_eval$specificity),
  Test_Precision = c(cl1_eval$precision, 
                     cl2_eval$precision, 
                     cl2_rna_eval$precision, 
                     cl1_img_eval$precision)
)

print(comparison_results)

comparison_results_rounded <- comparison_results
numeric_cols <- sapply(comparison_results_rounded, is.numeric)
comparison_results_rounded[numeric_cols] <- lapply(
  comparison_results_rounded[numeric_cols], 
  function(x) round(x, 4)
)

print(comparison_results_rounded)



#ROC curves
png("Output/mlp_roc_curves.png", width = 8, height = 8, units = "in", res = 300)
plot(1 - cl1_eval$specificity, cl1_eval$sensitivity, 
     type = "n",
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Figure 3: ROC Curves for Neural Network Models",
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)",
     cex.main = 1.8, cex.lab = 1.5, cex.axis = 1.3)
abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 1.5)
grid(col = "lightgray", lty = 1)
lines(1 - cl1_eval$roc_obj$specificities, cl1_eval$roc_obj$sensitivities, col = "#2E86AB", lwd = 2)
lines(1 - cl2_eval$roc_obj$specificities, cl2_eval$roc_obj$sensitivities, col = "#A23B72", lwd = 2)
lines(1 - cl2_rna_eval$roc_obj$specificities, cl2_rna_eval$roc_obj$sensitivities, col = "red", lwd = 2)
lines(1 - cl1_img_eval$roc_obj$specificities, cl1_img_eval$roc_obj$sensitivities, col = "blue", lwd = 2)
legend("bottomright", 
       legend = c(paste0("A: Clinical (AUC = ", sprintf("%.3f", cl1_eval$test_auc), ")"),
                  paste0("B: Clinical (AUC = ", sprintf("%.3f", cl2_eval$test_auc), ")"),
                  paste0("B: Clinical + Genomics (AUC = ", sprintf("%.3f", cl2_rna_eval$test_auc), ")"),
                  paste0("A: Clinical + Imaging (AUC = ", sprintf("%.3f", cl1_img_eval$test_auc), ")")),
       col = c("#2E86AB", "#A23B72", "red", "blue"),lwd = 3, cex = 1.1, pt.cex = 1.4, 
       bg = "white", box.col = "gray50", box.lwd = 1.5, inset = c(0.02, 0.02))
dev.off()

########CALIBRATION CURVES

#function to calculate calibration data
get_calibration_data <- function(predictions, y_test, n_bins = 10) {
  bins <- seq(0, 1, length.out = n_bins + 1)
  bin_means <- numeric(n_bins)
  obs_props <- numeric(n_bins)
  
  for(i in 1:n_bins) {
    in_bin <- predictions >= bins[i] & predictions < bins[i+1]
    if(sum(in_bin) > 0) {
      bin_means[i] <- mean(predictions[in_bin])
      obs_props[i] <- mean(y_test[in_bin])
    } else {
      bin_means[i] <- NA
      obs_props[i] <- NA
    }
  }
  
  valid <- !is.na(bin_means)
  return(data.frame(
    bin_mean = bin_means[valid],
    observed = obs_props[valid]
  ))
}

cal_cl1 <- get_calibration_data(cl1_eval$predictions, cl1_eval$y_test)
cal_cl2 <- get_calibration_data(cl2_eval$predictions, cl2_eval$y_test)
cal_cl2_rna <- get_calibration_data(cl2_rna_eval$predictions, cl2_rna_eval$y_test)
cal_cl1_img <- get_calibration_data(cl1_img_eval$predictions, cl1_img_eval$y_test)

plot(0, 0, type = "n", 
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Neural Network Comparison - Calibration Curves",
     xlab = "Mean Predicted Probability", 
     ylab = "Observed Proportion")
abline(0, 1, col = "gray", lwd = 2, lty = 2)

lines(cal_cl1$bin_mean, cal_cl1$observed, type = "b", col = "blue", lwd = 2, pch = 16)
lines(cal_cl2$bin_mean, cal_cl2$observed, type = "b", col = "red", lwd = 2, pch = 16)
lines(cal_cl2_rna$bin_mean, cal_cl2_rna$observed, type = "b", col = "green", lwd = 2, pch = 16)
lines(cal_cl1_img$bin_mean, cal_cl1_img$observed, type = "b", col = "purple", lwd = 2, pch = 16)

legend("bottomright", 
       legend = c("Perfect Calibration",
                  paste("CL1 (Brier:", round(cl1_eval$brier_score, 4), ")"),
                  paste("CL2 (Brier:", round(cl2_eval$brier_score, 4), ")"),
                  paste("CL2_RNA (Brier:", round(cl2_rna_eval$brier_score, 4), ")"),
                  paste("CL1_IMG (Brier:", round(cl1_img_eval$brier_score, 4), ")")),
       col = c("gray", "blue", "red", "green", "purple"),
       lwd = 2, lty = c(2, 1, 1, 1, 1),
       cex = 0.7)
grid()

dev.copy(png, "NN_Comparison_Calibration.png")
dev.off()


df_cl1 <- data.frame(
  pred = as.numeric(predict(cl1_tuning$final_model, 
                            as.matrix(predict(cl1_tuning$scale_params, cl1_reduced$x_test_reduced)), 
                            type = "raw")),actual = cl1_y_test,Model = "A: Clinical")

df_cl2 <- data.frame(
  pred = as.numeric(predict(cl2_tuning$final_model, 
                            as.matrix(predict(cl2_tuning$scale_params, cl2_reduced$x_test_reduced)), 
                            type = "raw")),actual = cl2_y_test,Model = "B: Clinical")

df_cl2_rna <- data.frame(
  pred = as.numeric(predict(cl2_rna_tuning$final_model, 
                            as.matrix(predict(cl2_rna_tuning$scale_params, cl2_rna_reduced$x_test_reduced)), 
                            type = "raw")),actual = cl2_rna_y_test,Model = "B: Clinical + Genomics")

df_cl1_img <- data.frame(
  pred = as.numeric(predict(cl1_img_tuning$final_model, 
                            as.matrix(predict(cl1_img_tuning$scale_params, cl1_img_reduced$x_test_reduced)), 
                            type = "raw")),actual = cl1_img_y_test,Model = "A: Clinical + Imaging")

combined_df <- rbind(df_cl1, df_cl2, df_cl2_rna, df_cl1_img)

png("Output/mlp_cal_curves_smooth.png", width = 10, height = 8, units = "in", res = 300)
plot(0, 0, type = "n", 
     xlim = c(0, 0.8), ylim = c(0, 1),
     main = "Figure 4: Calibration Curves for Neural Network Models",
     xlab = "Predicted Probability of Death",
     ylab = "Observed Proportion of Death",
     cex.main = 1.8, cex.lab = 1.5, cex.axis = 1.3)
abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 2)
grid(col = "lightgray", lty = 1)
add_smooth_curve <- function(pred, actual, color, model_name) {
  valid_idx <- complete.cases(pred, actual)
  pred <- pred[valid_idx]
  actual <- actual[valid_idx]
  loess_fit <- loess(actual ~ pred, degree = 1, span = 1.2)
  pred_sorted <- sort(pred)
  fit_smooth <- predict(loess_fit, newdata = data.frame(pred = pred_sorted), se = TRUE)
  lines(pred_sorted, fit_smooth$fit, col = color, lwd = 3)
  polygon(c(pred_sorted, rev(pred_sorted)), 
          c(fit_smooth$fit + 1.96 * fit_smooth$se, 
            rev(fit_smooth$fit - 1.96 * fit_smooth$se)),
          col = adjustcolor(color, alpha.f = 0.15), border = NA)
}
add_smooth_curve(df_cl1$pred, df_cl1$actual, "#2E86AB", "A: Clinical")
add_smooth_curve(df_cl2$pred, df_cl2$actual, "#A23B72", "B: Clinical")
add_smooth_curve(df_cl2_rna$pred, df_cl2_rna$actual, "#D62828", "B: Clinical + Genomics")
add_smooth_curve(df_cl1_img$pred, df_cl1_img$actual, "#003F88", "A: Clinical + Imaging")
legend("bottomright",
       legend = c("A: Clinical", "B: Clinical", 
                  "B: Clinical + Genomics", "A: Clinical + Imaging"),
       col = c("#2E86AB", "#A23B72", "#D62828", "#003F88"),
       lwd = 3, cex = 1.1, bg = "white", box.col = "gray50")
dev.off()


########SUMMARY TABLE

summary_nn <- data.frame(
  Dataset = c("CL1", "CL2", "CL2_RNA", "CL1_IMG"),
  AUC = round(c(cl1_eval$test_auc, cl2_eval$test_auc, 
                cl2_rna_eval$test_auc, cl1_img_eval$test_auc), 4),
  Brier = round(c(cl1_eval$brier_score, cl2_eval$brier_score, 
                  cl2_rna_eval$brier_score, cl1_img_eval$brier_score), 4),
  Accuracy = round(c(cl1_eval$accuracy, cl2_eval$accuracy, 
                     cl2_rna_eval$accuracy, cl1_img_eval$accuracy), 4),
  Sensitivity = round(c(cl1_eval$sensitivity, cl2_eval$sensitivity, 
                        cl2_rna_eval$sensitivity, cl1_img_eval$sensitivity), 4),
  Specificity = round(c(cl1_eval$specificity, cl2_eval$specificity, 
                        cl2_rna_eval$specificity, cl1_img_eval$specificity), 4))

print(summary_nn)



##conduct PFI (Permutation Feature Importance)


pfi <- function(x_train, y_train, x_test, y_test, final_model, scale_params, dataset_name) {
  
  x_test_scaled <- as.matrix(predict(scale_params, x_test))
  
  pred_baseline <- as.numeric(predict(final_model, x_test_scaled, type = "raw"))
  auc_baseline <- roc(y_test, pred_baseline, quiet = TRUE)$auc
  
  n_features <- ncol(x_test_scaled)
  feature_names <- colnames(x_test_scaled)
  importance_scores <- numeric(n_features)
  
  cat("Baseline AUC:", round(auc_baseline, 4), "\n\n")
  
  for(i in 1:n_features) {
    auc_permuted <- numeric(5)
    
    for(rep in 1:5) {
      x_permuted <- x_test_scaled
      x_permuted[, i] <- sample(x_permuted[, i])  
      pred_permuted <- as.numeric(predict(final_model, x_permuted, type = "raw"))
      auc_permuted[rep] <- roc(y_test, pred_permuted, quiet = TRUE)$auc
    }
    
    importance_scores[i] <- auc_baseline - mean(auc_permuted)
    
    cat(sprintf("%30s: AUC drop = %.4f\n", 
                substr(feature_names[i], 1, 30), 
                importance_scores[i]))
  }
  importance_df <- data.frame(
    Feature = feature_names,
    Importance = importance_scores
  ) %>%
    arrange(desc(Importance))
  
  print(head(importance_df, 5))
  
  return(importance_df)
}

cl1_importance <- pfi(cl1_reduced$x_train_reduced, cl1_y_train,
                             cl1_reduced$x_test_reduced, cl1_y_test,
                             cl1_tuning$final_model, cl1_tuning$scale_params,
                             "CL1")

cl2_importance <- pfi(cl2_reduced$x_train_reduced, cl2_y_train,
                             cl2_reduced$x_test_reduced, cl2_y_test,
                             cl2_tuning$final_model, cl2_tuning$scale_params,
                             "CL2")

cl2_rna_importance <- pfi(cl2_rna_reduced$x_train_reduced, cl2_rna_y_train,
                                 cl2_rna_reduced$x_test_reduced, cl2_rna_y_test,
                                 cl2_rna_tuning$final_model, cl2_rna_tuning$scale_params,
                                 "CL2_RNA")

cl1_img_importance <- pfi(cl1_img_reduced$x_train_reduced, cl1_img_y_train,
                                 cl1_img_reduced$x_test_reduced, cl1_img_y_test,
                                 cl1_img_tuning$final_model, cl1_img_tuning$scale_params,
                                 "CL1_IMG")

importance_results <- list(
  CL1 = cl1_importance,
  CL2 = cl2_importance,
  CL2_RNA = cl2_rna_importance,
  CL1_IMG = cl1_img_importance)


importance_summary <- function(imp_df, dataset_name) {
  imp_df %>%
    slice_head(n = 5) %>%
    mutate(Dataset = dataset_name)}

top_features <- bind_rows(
  importance_summary(cl1_importance, "CL1"),
  importance_summary(cl2_importance, "CL2"),
  importance_summary(cl2_rna_importance, "CL2_RNA"),
  importance_summary(cl1_img_importance, "CL1_IMG"))

print(top_features)

all_importance <- bind_rows(
  cl1_importance %>% head(5) %>% mutate(Dataset = "CL1"),
  cl2_importance %>% head(5) %>% mutate(Dataset = "CL2"),
  cl2_rna_importance %>% head(5) %>% mutate(Dataset = "CL2_RNA"),
  cl1_img_importance %>% head(5) %>% mutate(Dataset = "CL1_IMG"))

ggplot(all_importance, aes(x = reorder(Feature, Importance), y = Importance, fill = Dataset)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Dataset, scales = "free_y") +
  coord_flip() +
  labs(title = "Top 5 Features by Dataset",
       subtitle = "Permutation Feature Importance (Higher = More Important)",
       x = "Feature", 
       y = "AUC Drop") +
  theme_minimal() +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold"),
        plot.title = element_text(size = 14, face = "bold"))

ggsave("Combined_PFI_Comparison.png", width = 12, height = 8)


clean_feature_names <- function(feature_names, max_width = 25) {
  cleaned <- feature_names %>%
    gsub("_\\d+$", "", .) %>%
    gsub("gender_num", "Gender", .) %>%
    gsub("hist_num", "Histology", .) %>%
    gsub("ethnic_num", "Ethnicity", .) %>%
    gsub("XGG_num", "XGG", .) %>%
    gsub("path_t_num", "Path T Stage", .) %>%
    gsub("path_m_num", "Path M Stage", .) %>%
    gsub("path_n_num", "Path N Stage", .) %>%
    gsub("lymph_num", "Lymph Nodes", .) %>%
    gsub("egfr_num", "EGFR Status", .) %>%
    gsub("kras_num", "KRAS Status", .) %>%
    gsub("smoke_num", "Smoking Status", .) %>%
    gsub("T_stage", "T Stage", .) %>%
    gsub("N_stage", "N Stage", .) %>%
    gsub("M_stage", "M Stage", .) %>%
    gsub("overall_stage", "Overall Stage", .) %>%
    gsub("tl_", "Tumor Load ", .) %>%
    gsub("Age", "Age", .) %>%
    gsub("pleural", "Pleural Involvement", .) %>%
    gsub("adj_trt", "Adjuvant Treatment", .) %>%
    gsub("Chemotherapy", "Chemotherapy", .) %>%
    gsub("Radiation", "Radiation", .) %>%
    gsub("Recurrence", "Recurrence", .) %>%
    gsub("weight", "Weight", .) %>%
    gsub("pack_yrs", "Pack Years", .) %>%
    gsub("quit_smoke_yr", "Quit Smoking Years", .)
  
  wrapped <- str_wrap(cleaned, width = max_width)
  
  return(wrapped)
}

pfi <- function(x_train, y_train, x_test, y_test, final_model, scale_params, dataset_name) {
  
  x_test_scaled <- as.matrix(predict(scale_params, x_test))
  
  pred_baseline <- as.numeric(predict(final_model, x_test_scaled, type = "raw"))
  auc_baseline <- roc(y_test, pred_baseline, quiet = TRUE)$auc
  
  n_features <- ncol(x_test_scaled)
  feature_names <- colnames(x_test_scaled)
  importance_scores <- numeric(n_features)
  
  cat("Baseline AUC:", round(auc_baseline, 4), "\n")
  cat("Test set size:", nrow(x_test_scaled), "samples\n")
  cat("Features:", n_features, "\n\n")
  
  for(i in 1:n_features) {
    auc_permuted <- numeric(10)  
    
    for(rep in 1:10) {
      x_permuted <- x_test_scaled
      x_permuted[, i] <- sample(x_permuted[, i])
      pred_permuted <- as.numeric(predict(final_model, x_permuted, type = "raw"))
      auc_permuted[rep] <- roc(y_test, pred_permuted, quiet = TRUE)$auc
    }
    
    importance_scores[i] <- auc_baseline - mean(auc_permuted)
    sd_importance <- sd(auc_baseline - auc_permuted)
    
    if(i <= 10 || importance_scores[i] > 0.01) {
      cat(sprintf("  %-35s: AUC drop = %.4f (SD: %.4f)\n", 
                  substr(feature_names[i], 1, 35), 
                  importance_scores[i],
                  sd_importance))
    }
  }
  
  importance_df <- data.frame(
    Feature = feature_names,
    Importance = importance_scores,
    Feature_Cleaned = clean_feature_names(feature_names, max_width = 30)
  ) %>%
    arrange(desc(Importance))
  
  top10 <- head(importance_df, 10)
  for(i in 1:nrow(top10)) {
    cat(sprintf("%2d. %-40s: %.4f\n", 
                i, 
                top10$Feature_Cleaned[i], 
                top10$Importance[i]))
  }
  
  return(importance_df)
}

cl1_importance <- pfi(cl1_reduced$x_train_reduced, cl1_y_train,
                      cl1_reduced$x_test_reduced, cl1_y_test,
                      cl1_tuning$final_model, cl1_tuning$scale_params,
                      "A: Clinical")

cl2_importance <- pfi(cl2_reduced$x_train_reduced, cl2_y_train,
                      cl2_reduced$x_test_reduced, cl2_y_test,
                      cl2_tuning$final_model, cl2_tuning$scale_params,
                      "B: Clinical")

cl2_rna_importance <- pfi(cl2_rna_reduced$x_train_reduced, cl2_rna_y_train,
                          cl2_rna_reduced$x_test_reduced, cl2_rna_y_test,
                          cl2_rna_tuning$final_model, cl2_rna_tuning$scale_params,
                          "B: Clinical + Genomics")

cl1_img_importance <- pfi(cl1_img_reduced$x_train_reduced, cl1_img_y_train,
                          cl1_img_reduced$x_test_reduced, cl1_img_y_test,
                          cl1_img_tuning$final_model, cl1_img_tuning$scale_params,
                          "A: Clinical + Imaging")

all_importance <- bind_rows(
  cl1_importance %>% 
    head(10) %>% 
    mutate(Dataset = "A: Clinical",
           Dataset_Label = "A: Clinical (n=4 features)"),
  cl2_importance %>% 
    head(10) %>% 
    mutate(Dataset = "B: Clinical",
           Dataset_Label = paste0("B: Clinical (n=", nrow(cl2_importance), " features)")),
  cl2_rna_importance %>% 
    head(10) %>% 
    mutate(Dataset = "B: Clinical + Genomics",
           Dataset_Label = paste0("B: Clinical + Genomics (n=", nrow(cl2_rna_importance), " features)")),
  cl1_img_importance %>% 
    head(10) %>% 
    mutate(Dataset = "A: Clinical + Imaging",
           Dataset_Label = paste0("A: Clinical + Imaging (n=", nrow(cl1_img_importance), " features)"))
)


all_importance_top5 <- all_importance %>%
  group_by(Dataset) %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 5) %>% 
  ungroup() %>%
  group_by(Dataset) %>%
  mutate(Significant = Importance > (max(Importance) * 0.1))


plot3_top5_minimal_white <- ggplot(all_importance_top5, 
                                   aes(x = reorder(Feature_Cleaned, Importance), 
                                       y = Importance, 
                                       fill = Significant)) +
  geom_bar(stat = "identity", alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", Importance)), 
            hjust = -0.2, size = 3.5, color = "black") +
  scale_fill_manual(values = c("TRUE" = "coral3", "FALSE" = "steelblue"),
                    labels = c("TRUE" = "Important", "FALSE" = "Less Important")) +
  scale_y_continuous(expand = expansion(mult = c(0.01, 0.15))) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
  facet_wrap(~Dataset_Label, scales = "free_y", ncol = 2) +
  coord_flip(clip = "off") +
  labs(title = "Figure 5: Top 5 Features by Model Type",
       subtitle = "Permutation Feature Importance: Higher AUC Drop Indicates Greater Importance",
       x = "", 
       y = "AUC Drop (Importance Score)",
       fill = "Significance") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    strip.text = element_text(face = "bold", size = 11),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = margin(b = 5)),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30", margin = margin(b = 10)),
    axis.text.y = element_text(size = 10, hjust = 1),
    axis.text.x = element_text(size = 9),
    axis.title.x = element_text(margin = margin(t = 10)),
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),  # Keep horizontal grid
    panel.grid.major.x = element_blank(),  # Remove vertical grid
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.2, "lines"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(10, 20, 10, 10)
  )

print(plot3_top5_minimal_white)

ggsave("Output/pfi_importance_top5_minimal_white.png", 
       plot3_top5_minimal_white, 
       width = 12, 
       height = 8, 
       dpi = 300,
       bg = "white")


