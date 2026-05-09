#############################################################################
#                                                                           #
# Program Name:  Data processing - images                                   #
#                                                                           #
# Outputs:      cl1_img_train_imputed, cl1_img_test_imputed                 #
#               cl1_img_scl_train_imputed, cl1_img_scl_test_imputed         #
#                                                                           #
#                                                                           #
#############################################################################

#install.packages("")

###############################
##      LOAD PACKAGES       ##
##############################
library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)
library(VIM)


## set working directory ##

setwd("C:/Users/phil0068/DataNow/Home/MASTERS/AI_in_HDS/Summative")

#####  READ IN DATA   ##########

shape <- read.csv("Data/Raw/features_shape.csv")
intensity <- read.csv("Data/Raw/features_intensity.csv")
texture <- read.csv("Data/Raw/features_texture.csv")
cl1_clean <- read.csv("Data/Derived/clinical1_clean.csv")

#log from python script run via HPC shows that 6 patients did not have image data available
#resulting in 416 patients with image data
#1 pt with missing target will be removed


#Join features into one dataset
img <- shape |>
  full_join(intensity, by = "patient_id") |>
  full_join(texture, by = "patient_id")

#shape - 6 features
#intensity - 6 features
#texture - 3 features
#img - 15 features + patient ID

#scale energy to be more interpretable - currently very small values

img$energy_scaled <- img$energy * 100000

#nothing more to do process for using this data in XGBoost - save to csv.

write.csv(img, "Data/Derived/img.csv")

#merge image data onto clinical 1 data
cl1_img <- img |>
  left_join(
    cl1_clean |> select(-c(surv,cens,surv_3y,cens_3y,X)), #drop other tagret vars
    by = c("patient_id" = "ID")) |>
  rename(ID=patient_id) |>
  filter(!is.na(status_3y)) #remove any patients with missing target (1pt)

write.csv(cl1_img, "Data/Derived/cl1_img.csv") #currently missingness in features included

##### now preprocess for use in the neural networks  #######

#Check distribution of features
img_long <- img |>
  select(-patient_id) |>
  pivot_longer(everything(), names_to = "feature", values_to = "value")

ggplot(img_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  facet_wrap(~feature, scales = "free") +
  theme_minimal() +
  ggtitle("Distribution of All Features")
ggplot(img_long, aes(x = value, fill = feature)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~feature, scales = "free") +
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Density Plots of All Features")

img_homo <- img |>
  mutate(homogeneity_spread = 1 - homogeneity)

#compactness, kurtosis, volume are right skewed
#std_intensity, max_diameter are left skewed
#cannot transform kurtosis because vlaues are negative

#log transform skewed features
img_log <- img_homo |>
  mutate(
    log_volume = log(volume),
    log_compactness = log(compactness),
    log_std_intensity = log(std_intensity),
    log_max_diameter = log(max_diameter))

#check for outliers
boxplot(img$volume)
boxplot(img$elongation) #2 outliers
boxplot(img$energy_scaled)
boxplot(img$entropy) #2 outliers
boxplot(img$kurtosis)
boxplot(img$max_diameter) #4 outliers
boxplot(img$sphericity)


#normalise features
#z-score
img_z <- img_log |>
  mutate(across(c(log_volume, surface_area, log_max_diameter, sphericity, 
                  log_compactness, elongation, mean_intensity, log_std_intensity,
                  skewness, kurtosis, entropy, contrast, dissimilarity, homogeneity),
                ~ as.numeric(scale(.)))) |>
  select(-c(energy,volume,compactness,std_intensity,max_diameter))


#check correlation between features
numeric_cols <- img_z |>
  select(-patient_id)

#correlation matrix
cor_matrix <- cor(numeric_cols)

#check the matrix
round(cor_matrix, 2)

#find correlated variables and identify which to remove
high_cor <- findCorrelation(cor_matrix, cutoff = 0.85, verbose = TRUE)

columns_to_remove <- colnames(cor_matrix)[high_cor]
print(columns_to_remove)

img_z_nocor <- img_z |>
  select(-c(dissimilarity,homogeneity,mean_intensity,log_compactness))

#merge image data onto clinical 1 data
cl1_img_scaled <- img_z_nocor |>
  left_join(
    cl1_clean |> select(-c(surv,cens,surv_3y,cens_3y,X)), #drop other target vars
    by = c("patient_id" = "ID")) |>
  rename(ID=patient_id) |> 
  filter(!is.na(status_3y)) #remove any patients with missing target (1pt)

write.csv(cl1_img_scaled, "Data/Derived/cl1_img_scaled.csv") # currently includes missingness in features


########### Split data into Train and Test ###########

#need to split all data (all splits need to be the same), stratified by the target:
#cl1_img, cl1_img_scaled


#####################################################
# cl1_img - clinical 1 with images - no scaling or processing # 
#####################################################

set.seed(123)
cl1_img_indices <- createDataPartition(cl1_img$status_3y, p = 0.8, list = FALSE)

cl1_img_train <- cl1_img[cl1_img_indices, ]
cl1_img_test <- cl1_img[-cl1_img_indices, ]

#Verify proportions
cat("Train set size:", nrow(cl1_img_train), "\n")
cat("Test set size:", nrow(cl1_img_test), "\n")
cat("Proportion of events in train:", 
    mean(cl1_img_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl1_img_test$status_3y == 1, na.rm = TRUE), "\n")
#happy


#####################################################
# cl1_img_scaled - clinical 1 with images - WITH scaling or processing # 
#####################################################

set.seed(123)
cl1_img_scaled_indices <- createDataPartition(cl1_img_scaled$status_3y, p = 0.8, list = FALSE)

cl1_img_scaled_train <- cl1_img[cl1_img_scaled_indices, ]
cl1_img_scaled_test <- cl1_img[-cl1_img_scaled_indices, ]

#Verify proportions
cat("Train set size:", nrow(cl1_img_scaled_train), "\n")
cat("Test set size:", nrow(cl1_img_scaled_test), "\n")
cat("Proportion of events in train:", 
    mean(cl1_img_scaled_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl1_img_scaled_test$status_3y == 1, na.rm = TRUE), "\n")
#happy

#save split datasets
write.csv(cl1_img_train, "Data/Derived/cl1_img_train.csv")
write.csv(cl1_img_test, "Data/Derived/cl1_img_test.csv")
write.csv(cl1_img_scaled_train, "Data/Derived/cl1_img_scaled_train.csv")
write.csv(cl1_img_scaled_test, "Data/Derived/cl1_img_scaled_test.csv")


####Now, apply K-NN (k=3) to the missingness approach datasets

#Impute with kNN (fits on training set, then is applied to both train and test)

set.seed(123)
#Impute clinical 1 and image date non scaled
cl1_img_train_imputed_pre <- kNN(cl1_img_train[, !names(cl1_img_train) %in% c("status_3y")], k = 3)
cl1_img_train_imputed_clean <- cl1_img_train_imputed_pre[, !grepl("_imp$", names(cl1_img_train_imputed_pre))]
cl1_img_train_imputed <- cbind(cl1_img_train_imputed_clean,status_3y = cl1_img_train[, "status_3y"])

cl1_img_test_imputed_pre <- kNN(cl1_img_test[, !names(cl1_img_test) %in% c("status_3y")], k = 3)
cl1_img_test_imputed_clean <- cl1_img_test_imputed_pre[, !grepl("_imp$", names(cl1_img_test_imputed_pre))]
cl1_img_test_imputed <- cbind(cl1_img_test_imputed_clean,status_3y = cl1_img_test[, "status_3y"])

set.seed(123)
#Impute clinical 1 and image data scaled
cl1_img_scl_train_imputed_pre <- kNN(cl1_img_scaled_train[, !names(cl1_img_scaled_train) %in% c("status_3y")], k = 3)
cl1_img_scl_train_imputed_clean <- cl1_img_scl_train_imputed_pre[, !grepl("_imp$", names(cl1_img_scl_train_imputed_pre))]
cl1_img_scl_train_imputed <- cbind(cl1_img_scl_train_imputed_clean,status_3y = cl1_img_scaled_train[, "status_3y"])

cl1_img_scl_test_imputed_pre <- kNN(cl1_img_scaled_test[, !names(cl1_img_scaled_test) %in% c("status_3y")], k = 3)
cl1_img_scl_test_imputed_clean <- cl1_img_scl_test_imputed_pre[, !grepl("_imp$", names(cl1_img_scl_test_imputed_pre))]
cl1_img_scl_test_imputed <- cbind(cl1_img_scl_test_imputed_clean,status_3y = cl1_img_scaled_test[, "status_3y"])


#save imputed datasets
write.csv(cl1_img_train_imputed, "Data/Derived/cl1_img_train_imputed")
write.csv(cl1_img_test_imputed, "Data/Derived/cl1_img_test_imputed")
write.csv(cl1_img_scl_train_imputed, "Data/Derived/cl1_img_scl_train_imputed")
write.csv(cl1_img_scl_test_imputed, "Data/Derived/cl1_img_scl_test_imputed")


