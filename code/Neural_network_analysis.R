#############################################################################
#                                                                           #
# Program Name:  Neural Network Analysis                                           #
#                                                                           #
#############################################################################


###############################
##      LOAD PACKAGES       ##
##############################
#library(ggplot2)
library(glmnet)
library(pROC)
library(caret)
#library(keras3)
#library(tensorflow)
#library(dplyr)


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
  
  # Run lasso CV
  cv_lasso <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = 1,
    type.measure = "auc",
    nfolds = 5
  )
  
  # Extract features with non-zero coefficients at lambda.1se
  coef_matrix <- as.matrix(coef(cv_lasso, s = cv_lasso$lambda.1se))
  selected_features <- rownames(coef_matrix)[coef_matrix[, 1] != 0]
  selected_features <- selected_features[selected_features != "(Intercept)"]
  
  # Get coefficients
  feature_coefs <- coef_matrix[selected_features, 1]
  
  # Summary
  cat("  Features:", length(selected_features), "/", ncol(x_train), 
      "| Max AUC:", round(max(cv_lasso$cvm), 4), "\n")
  
  if(length(selected_features) > 0) {
    # Show top 5
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




#EVALUATE LASSO ON TEST SETS

evaluate_lasso_test <- function(x_train, y_train, x_test, y_test, selected_features, dataset_name) {
  
  cat("\n", paste(rep("=", 50), collapse = ""), "\n")
  cat(dataset_name, "LASSO EVALUATION ON TEST SET\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  
  available_features <- intersect(selected_features, colnames(x_test))
  available_features <- intersect(available_features, colnames(x_train))
  
  if(length(available_features) == 0) {
    cat("ERROR: No common features found!\n")
    return(NULL)
  }
  
  x_train_subset <- x_train[, available_features, drop = FALSE]
  x_test_subset <- x_test[, available_features, drop = FALSE]
  
  cat("Features used:", length(available_features), "\n")
  cat("Test set size:", nrow(x_test_subset), "samples\n")
  
  cv_lasso <- cv.glmnet(
    x = x_train_subset,
    y = y_train,
    family = "binomial",
    alpha = 1,
    type.measure = "auc",
    nfolds = min(5, nrow(x_train_subset))
  )
  
  test_pred <- predict(cv_lasso, newx = x_test_subset, s = cv_lasso$lambda.1se, type = "response")
  
  test_pred <- as.numeric(test_pred)
  
  test_auc <- auc(roc(y_test, test_pred, quiet = TRUE))
  
  roc_obj <- roc(y_test, test_pred, quiet = TRUE)
  optimal_threshold <- coords(roc_obj, "best", ret = "threshold", transpose = FALSE)[1]
  optimal_threshold <- as.numeric(optimal_threshold)
  
  predicted_class <- ifelse(test_pred > optimal_threshold, 1, 0)
  
  accuracy <- mean(predicted_class == y_test)
  
  tp <- sum(predicted_class == 1 & y_test == 1)
  tn <- sum(predicted_class == 0 & y_test == 0)
  fp <- sum(predicted_class == 1 & y_test == 0)
  fn <- sum(predicted_class == 0 & y_test == 1)
  
  sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), NA)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), NA)
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), NA)
  f1 <- ifelse((precision + sensitivity) > 0, 2 * precision * sensitivity / (precision + sensitivity), NA)
  
  cat("\nResults (using optimal threshold =", round(optimal_threshold, 4), "):\n")
  cat("Test AUC:", round(test_auc, 4), "\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
  cat("Sensitivity:", round(sensitivity, 4), "\n")
  cat("Specificity:", round(specificity, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("F1 Score:", round(f1, 4), "\n")
  
  cat("\nConfusion Matrix:\n")
  cat("            Predicted\n")
  cat("Actual    0    1\n")
  cat("    0    ", tn, "  ", fp, "\n")
  cat("    1    ", fn, "  ", tp, "\n")
  
  return(list(
    test_auc = test_auc,
    predictions = test_pred,
    predicted_class = predicted_class,
    optimal_threshold = optimal_threshold,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1 = f1,
    features_used = available_features
  ))
}

# Run for all datasets
cl1_eval <- evaluate_lasso_test(cl1_x_train, cl1_y_train, cl1_x_test, cl1_y_test, 
                                         cl1_lasso$selected_features, "CL1")

cl2_eval <- evaluate_lasso_test(cl2_x_train, cl2_y_train, cl2_x_test, cl2_y_test, 
                                         cl2_lasso$selected_features, "CL2")

cl2_rna_eval <- evaluate_lasso_test(cl2_rna_x_train, cl2_rna_y_train, 
                                             cl2_rna_x_test, cl2_rna_y_test, 
                                             cl2_rna_lasso$selected_features, "CL2_RNA")

cl1_img_eval <- evaluate_lasso_test(cl1_img_x_train, cl1_img_y_train, 
                                             cl1_img_x_test, cl1_img_y_test, 
                                             cl1_img_lasso$selected_features, "CL1_IMG")

# Compare all results
comparison_results <- data.frame(
  Dataset = c("CL1", "CL2", "CL2_RNA", "CL1_IMG"),
  AUC = c(cl1_eval$test_auc, cl2_eval$test_auc, cl2_rna_eval$test_auc, cl1_img_eval$test_auc),
  Accuracy = c(cl1_eval$accuracy, cl2_eval$accuracy, cl2_rna_eval$accuracy, cl1_img_eval$accuracy),
  Sensitivity = c(cl1_eval$sensitivity, cl2_eval$sensitivity, cl2_rna_eval$sensitivity, cl1_img_eval$sensitivity),
  Specificity = c(cl1_eval$specificity, cl2_eval$specificity, cl2_rna_eval$specificity, cl1_img_eval$specificity),
  Features_Used = c(length(cl1_eval$features_used), length(cl2_eval$features_used),
                    length(cl2_rna_eval$features_used), length(cl1_img_eval$features_used))
)

print(paste(rep("=", 60), collapse = ""))
print(comparison_results)



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

# Improved version with small fixes
tune_mlp_r <- function(x_train, y_train, n_folds = 4, n_trials = 20) {
  
  set.seed(123)
  
  # Convert to matrix if needed
  x_train <- as.matrix(x_train)
  
  folds <- createFolds(y_train, k = n_folds, list = TRUE)
  
  best_auc <- 0
  best_params <- NULL
  results <- data.frame()
  
  cat("Starting PURE R MLP tuning (nnet)...\n")
  cat("Using", n_folds, "fold CV\n")
  cat("Samples:", nrow(x_train), " Features:", ncol(x_train), "\n\n")
  
  for(i in 1:n_trials) {
    
    # Random search space (adjusted for your small dataset)
    params <- list(
      size = sample(c(1, 2, 3, 4, 5, 6), 1),     # Smaller max for your 4 features
      decay = 10^runif(1, -4, -1),                # L2 regularization
      maxit = sample(c(200, 300, 400, 500), 1)
    )
    
    fold_aucs <- c()
    
    # Cross validation
    for(fold in 1:n_folds) {
      
      val_idx <- folds[[fold]]
      train_idx <- setdiff(seq_len(nrow(x_train)), val_idx)
      
      # Scale inside fold (critical for neural nets)
      scale_params <- preProcess(x_train[train_idx, ], method = c("center", "scale"))
      x_train_fold <- predict(scale_params, x_train[train_idx, ])
      x_val_fold <- predict(scale_params, x_train[val_idx, ])
      
      # Train model
      model <- nnet(
        x = as.matrix(x_train_fold),
        y = y_train[train_idx],
        size = params$size,
        decay = params$decay,
        maxit = params$maxit,
        trace = FALSE,
        linout = FALSE  # For classification
      )
      
      # Predict on validation fold
      pred <- predict(model, as.matrix(x_val_fold), type = "raw")
      
      # Calculate AUC
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
    
    # Update best
    if(mean_auc > best_auc) {
      best_auc <- mean_auc
      best_params <- params
      
      cat(sprintf("Trial %d/%d -> NEW BEST CV AUC: %.4f (size=%d, decay=%.5f)\n", 
                  i, n_trials, mean_auc, params$size, params$decay))
    } else if(i %% 5 == 0) {
      cat(sprintf("Trial %d/%d -> Best so far: %.4f\n", i, n_trials, best_auc))
    }
  }
  
  # Train final model on ALL training data
  cat("\nTraining final model on full training set...\n")
  
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



#EVALUATE MODEL ON THE TEST SET USING MLP

evaluate_final_nnet <- function(tuning_result, x_test, y_test, dataset_name) {
  
  # Scale test set using training parameters
  x_test_scaled <- as.matrix(predict(tuning_result$scale_params, x_test))
  
  # Predict and convert to vector
  predictions <- as.numeric(predict(tuning_result$final_model, x_test_scaled, type = "raw"))
  
  # AUC-ROC
  roc_obj <- roc(y_test, predictions, quiet = TRUE)
  test_auc <- auc(roc_obj)
  
  # Brier score
  brier_score <- mean((predictions - y_test)^2)
  
  # Optimal threshold and classification metrics
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
  
  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat(dataset_name, "FINAL MODEL TEST PERFORMANCE\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
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

# Run evaluations
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


#ROC curves
plot(cl1_eval$roc_obj,col = "#2E86AB",lwd = 2,
     main = "Neural Network Comparison - ROC Curves",
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)")
lines(cl2_eval$roc_obj,col = "#A23B72",lwd = 2)
lines(cl2_rna_eval$roc_obj,col = "red",lwd = 2)
lines(cl1_img_eval$roc_obj,col = "blue",lwd = 2)
legend("bottomright",legend = c(
    paste("Clinical Model 1 (AUC =",round(cl1_eval$test_auc, 3), ")"),
    paste("Clinical Model 2 (AUC =",round(cl2_eval$test_auc, 3), ")"),
    paste("Clinical & RNA Model 3 (AUC =",round(cl2_rna_eval$test_auc, 3), ")"),
    paste("Clinical & Image Model 4 (AUC =",round(cl1_img_eval$test_auc, 3), ")")),
  col = c("#2E86AB", "#A23B72", "red", "blue"),lwd = 2,cex = 0.8,bg = "white",box.col = "gray80")
grid()

dev.copy(png, "Output/NN_Comparison_ROC_curves2.png")
dev.off()


library(ggplot2)
library(scales)

# Create dataframe for ggplot
roc_df <- rbind(
  data.frame(specificity = cl1_eval$roc_obj$specificities,
    sensitivity = cl1_eval$roc_obj$sensitivities,Model = "Clinical Model 1"),
  data.frame(specificity = cl2_eval$roc_obj$specificities,
    sensitivity = cl2_eval$roc_obj$sensitivities,Model = "Clinical Model 2"),
  data.frame(specificity = cl2_rna_eval$roc_obj$specificities,
    sensitivity = cl2_rna_eval$roc_obj$sensitivities,Model = "Clinical & RNA Model 3"),
  data.frame(specificity = cl1_img_eval$roc_obj$specificities,
    sensitivity = cl1_img_eval$roc_obj$sensitivities,Model = "Clinical & Image Model 4"))

# Plot ROC curves
ggplot(roc_df,aes(x = 1 - specificity,y = sensitivity,color = Model)) +
  geom_line(linewidth = 1.2) + geom_abline(linetype = "dashed", color = "gray50", alpha = 0.7) +
  scale_color_manual(
    values = c(
      "Clinical Model 1" = "#2E86AB",
      "Clinical Model 2" = alpha("#A23B72", 0.6),
      "Clinical & RNA Model 3" = "red",
      "Clinical & Image Model 4" = alpha("blue", 0.4)),
    labels = c(paste0("Clinical Model 1 (AUC = ", round(cl1_eval$test_auc, 3), ")"),
      paste0("Clinical Model 2 (AUC = ", round(cl2_eval$test_auc, 3), ")"),
      paste0("Clinical & RNA Model 3 (AUC = ", round(cl2_rna_eval$test_auc, 3), ")"),
      paste0("Clinical & Image Model 4 (AUC = ", round(cl1_img_eval$test_auc, 3), ")"))) +
  labs(x = "1 - Specificity (False Positive Rate)",y = "Sensitivity (True Positive Rate)",
    title = "ROC Curves for Neural Network Models",color = "Model") +
  theme_minimal() +theme(legend.position = c(0.75, 0.25), legend.background = element_rect(
    fill = "white", color = "gray80", linewidth = 0.5),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)) +
  coord_equal()


# ============================================
# CALIBRATION CURVES
# ============================================

# Function to calculate calibration data
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

# Get calibration data for each dataset
cal_cl1 <- get_calibration_data(cl1_eval$predictions, cl1_eval$y_test)
cal_cl2 <- get_calibration_data(cl2_eval$predictions, cl2_eval$y_test)
cal_cl2_rna <- get_calibration_data(cl2_rna_eval$predictions, cl2_rna_eval$y_test)
cal_cl1_img <- get_calibration_data(cl1_img_eval$predictions, cl1_img_eval$y_test)

# Plot calibration curves
plot(0, 0, type = "n", 
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Neural Network Comparison - Calibration Curves",
     xlab = "Mean Predicted Probability", 
     ylab = "Observed Proportion")
abline(0, 1, col = "gray", lwd = 2, lty = 2)

# Add lines
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

# Save
dev.copy(png, "NN_Comparison_Calibration.png")
dev.off()

# ============================================
# SUMMARY TABLE
# ============================================

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


# ============================================
# BAR PLOT COMPARISON
# ============================================

library(ggplot2)
library(tidyr)
library(dplyr)

# Reshape for plotting
summary_long <- summary_nn %>%
  select(Dataset, AUC, Brier, Accuracy) %>%
  pivot_longer(cols = c(AUC, Brier, Accuracy), 
               names_to = "Metric", 
               values_to = "Value")

ggplot(summary_long, aes(x = Dataset, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Neural Network Performance Comparison",
       y = "Score") +
  theme_minimal() +
  scale_fill_manual(values = c("AUC" = "steelblue", 
                               "Brier" = "orange", 
                               "Accuracy" = "green")) +
  coord_cartesian(ylim = c(0, 1))

ggsave("Output/NN_Performance_Barplot.png", width = 10, height = 6)


## conduct PFI (Permutation Feature Importance)


# Manual PFI implementation (no extra packages)
pfi <- function(x_train, y_train, x_test, y_test, final_model, scale_params, dataset_name) {
  
  # Scale test data
  x_test_scaled <- as.matrix(predict(scale_params, x_test))
  
  # Get baseline predictions and AUC
  pred_baseline <- as.numeric(predict(final_model, x_test_scaled, type = "raw"))
  auc_baseline <- roc(y_test, pred_baseline, quiet = TRUE)$auc
  
  # Calculate importance for each feature
  n_features <- ncol(x_test_scaled)
  feature_names <- colnames(x_test_scaled)
  importance_scores <- numeric(n_features)
  
  cat("\n", paste(rep("=", 50), collapse = ""), "\n")
  cat(dataset_name, "- Permutation Feature Importance\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  cat("Baseline AUC:", round(auc_baseline, 4), "\n\n")
  
  for(i in 1:n_features) {
    # Permute feature i multiple times for stability
    auc_permuted <- numeric(5)
    
    for(rep in 1:5) {
      x_permuted <- x_test_scaled
      x_permuted[, i] <- sample(x_permuted[, i])  # Shuffle feature i
      pred_permuted <- as.numeric(predict(final_model, x_permuted, type = "raw"))
      auc_permuted[rep] <- roc(y_test, pred_permuted, quiet = TRUE)$auc
    }
    
    # Importance = drop in AUC (baseline - permuted)
    importance_scores[i] <- auc_baseline - mean(auc_permuted)
    
    cat(sprintf("%30s: AUC drop = %.4f\n", 
                substr(feature_names[i], 1, 30), 
                importance_scores[i]))
  }
  
  # Create data frame
  importance_df <- data.frame(
    Feature = feature_names,
    Importance = importance_scores
  ) %>%
    arrange(desc(Importance))
  
  # Print top features
  cat("\n", paste(rep("-", 30), collapse = ""), "\n")
  cat("TOP 5 MOST IMPORTANT FEATURES\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  print(head(importance_df, 5))
  
  return(importance_df)
}

# Apply to your models
# Run PFI for all datasets
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
  CL1_IMG = cl1_img_importance
)


# Create a comparison summary
importance_summary <- function(imp_df, dataset_name) {
  imp_df %>%
    slice_head(n = 5) %>%
    mutate(Dataset = dataset_name)
}

# Combine top features from all datasets
top_features <- bind_rows(
  importance_summary(cl1_importance, "CL1"),
  importance_summary(cl2_importance, "CL2"),
  importance_summary(cl2_rna_importance, "CL2_RNA"),
  importance_summary(cl1_img_importance, "CL1_IMG")
)

print(top_features)

# Professional horizontal bar chart for report
plot_pfi_report <- function(importance_df, dataset_name, top_n = 10) {
  
  # Take top N features
  plot_df <- head(importance_df, top_n)
  
  # Calculate % of max importance for context
  plot_df$Relative_Importance <- (plot_df$Importance / max(plot_df$Importance)) * 100
  
  ggplot(plot_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
    geom_text(aes(label = round(Importance, 3), hjust = -0.2), size = 3) +
    coord_flip() +
    labs(title = paste("Feature Importance -", dataset_name),
         subtitle = "Permutation Feature Importance (Higher = More Important)",
         x = "Feature", 
         y = "Loss in AUC (Permutation Importance)") +
    theme_minimal() +
    theme(panel.grid.major.y = element_blank(),
          plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"),
          axis.text.y = element_text(size = 10))
}

# Generate for all datasets
plot_pfi_report(cl1_importance, "CL1", top_n = 4)  # Only 4 features total
plot_pfi_report(cl2_rna_importance, "CL2_RNA", top_n = 10)



# Combine all importance data for faceted plot
all_importance <- bind_rows(
  cl1_importance %>% head(5) %>% mutate(Dataset = "CL1"),
  cl2_importance %>% head(5) %>% mutate(Dataset = "CL2"),
  cl2_rna_importance %>% head(5) %>% mutate(Dataset = "CL2_RNA"),
  cl1_img_importance %>% head(5) %>% mutate(Dataset = "CL1_IMG")
)

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