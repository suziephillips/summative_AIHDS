#############################################################################
#                                                                           #
# Program Name:  Data processing v0.1                                       #
#                                                                           #
# Outputs:      
#
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
#library(RANN)
library(VIM)


## set working directory ##

setwd("C:/Users/phil0068/DataNow/Home/MASTERS/AI_in_HDS/Summative")

#####  READ IN DATA   ##########

cl1 <- read.csv("Data/Raw/clinical1.csv")
cl2 <- read.csv("Data/Raw/clinical2.csv")
#rna <- read.csv("Data/Raw/rnaseq.txt")
rna <- read.delim("Data/Raw/rnaseq.txt", header = TRUE)


##### Clean cl1 ######

#factorise each string categorical feature:

cl1$gender_num <- factor(cl1$gender, levels = c("male", "female"))
cl1$overall_stage_num <- as.numeric(factor(cl1$Overall.Stage, 
                                      levels = c("I","II","IIIa","IIIb")))
cl1$hist_num <- factor(cl1$Histology, levels = c("adenocarcinoma","squamous cell carcinoma","large cell"))
#'nos' taken to be 'not otherwise specified' to be missing data

#amend age to be to one decimal place
cl1$age_1dp <- round(cl1$age, digits=1)

#create survival variables
cl1 <- cl1 |>
  mutate(
    surv = Survival.time,
    cens = deadstatus.event)

#create binary survival status from a chosen fixed time point - 3 years
cl1 <- cl1 |>
  mutate(
    surv_3y = if_else(surv > 1095, 1095, surv),
    cens_3y = case_when(
      surv > 1095 & cens == 1 ~ 0,
      TRUE ~ cens),
    status_3y = case_when(
      cens_3y == 1 ~ 1,
      cens_3y == 0 & surv_3y == 1095 ~ 0,
      cens_3y == 0 & surv_3y < 1095 ~ NA_real_))
#2 instances where pt was lost to follow up before 3 years, so target is NA.

#keep only variables we need
cl1b <- cl1 |>
  select(-c(gender,Histology,age,Overall.Stage,Survival.time,deadstatus.event)) |>
  rename(ID = PatientID,
         T_stage = clinical.T.Stage,
         N_stage = Clinical.N.Stage,
         M_stage = Clinical.M.Stage,
         age = age_1dp,
         overall_stage = overall_stage_num) |>
  relocate(ID, status_3y, surv, cens, surv_3y, cens_3y, gender_num, age, hist_num, T_stage, N_stage, M_stage, overall_stage)

#check distribution of continuous vars
#ggplot(cl1b, aes(x = age)) + geom_histogram(bins = 30) #reasonable
#ggplot(cl1b, aes(x = age)) + geom_density() #reasonable

#standardise continuous data:
cl1c <- cl1b
cl1c$age_z <- as.numeric(scale(cl1b$age))


##### Clean cl2 #####

# Replaces all instances of missingness with NA in the entire data frame
cl2[cl2 == 'Not Collected'] <- NA
cl2[cl2 == 'Not collected'] <- NA
cl2[cl2 == 'Not Recorded In Database'] <- NA
cl2[cl2 == 'N/A'] <- NA
cl2[cl2 == 'Not Assessed'] <- NA
cl2[cl2 == 'Unknown'] <- NA

#factorise each string categorical feature:

cl2$site_num <- factor(cl2$Patient.affiliation, levels = c("Standford", "VA"))
cl2$gender_num <- factor(cl2$Gender, levels = c("Male", "Female"))
cl2$hist_num <- factor(cl2$Histology, levels = c("Adenocarcinoma","Squamous cell carcinoma"))
#'nos' taken to be 'not otherwise specified' to be missing data
cl2$ethnic_num <- factor(cl2$Ethnicity, levels = c("African-American","Asian","Caucasian","Hispanic/Latino","Native Hawaiian/Pacific Islander"))
cl2$XGG_num <- factor(cl2$X.GG, levels = c("0%",">0 - 25%","25 - 50%","50 - 75%","75 - < 100%","100%"))
cl2$path_t_num <- factor(cl2$Pathological.T.stage, levels = c("T1a","T1b","T2a","T2b","T3","T4","Tis"))
cl2$path_m_num <- factor(cl2$Pathological.M.stage, levels = c("M0","M1a","M1b"))
cl2$path_n_num <- factor(cl2$Pathological.N.stage, levels = c("N0","N1","N2"))
cl2$hist_num <- factor(cl2$Histopathological.Grade, levels = c("G1 Well differentiated","G2 Moderately differentiated","G3 Poorly differentiated","Other, Type I: Well to moderately differentiated","Other, Type II: Moderately to poorly differentiated"))
cl2$lymph_num <- factor(cl2$Lymphovascular.invasion, levels = c("Absent","Present"))
cl2$egfr_num <- factor(cl2$EGFR.mutation.status, levels = c("Mutant","Wildtype"))
cl2$kras_num <- factor(cl2$KRAS.mutation.status, levels = c("Mutant","Wildtype"))
cl2$alk_num <- factor(cl2$ALK.translocation.status, levels = c("Translocated","Wildtype"))
cl2$recur_loc_num <- factor(cl2$Recurrence.Location, levels = c("distant","local","regional"))
cl2$smoke_num <- factor(cl2$Smoking.status, levels = c("Nonsmoker","Former","Current"))

#numerise binary categorical variables
cl2 <- cl2 |>
  mutate(Survival.Status = case_when(
    Survival.Status == "Alive" ~ 0,
    Survival.Status == "Dead" ~ 1)) |>
  mutate(Chemotherapy = case_when(
    Chemotherapy == "No" ~ 0,
    Chemotherapy == "Yes" ~ 1)) |>
  mutate(Radiation = case_when(
    Radiation == "No" ~ 0,
    Radiation == "Yes" ~ 1)) |>
  mutate(Adjuvant.Treatment = case_when(
    Adjuvant.Treatment == "No" ~ 0,
    Adjuvant.Treatment == "Yes" ~ 1)) |>
  mutate(Recurrence = case_when(
    Recurrence == "no" ~ 0,
    Recurrence == "yes" ~ 1)) |>
  mutate(Pleural.invasion..elastic..visceral..or.parietal. = case_when(
    Pleural.invasion..elastic..visceral..or.parietal. == "No" ~ 0,
    Pleural.invasion..elastic..visceral..or.parietal. == "Yes" ~ 1)) |>
  rename(pleural = Pleural.invasion..elastic..visceral..or.parietal.)|>
  mutate(across(starts_with("Tumor.Location"), ~ case_when(
                . == "Checked"   ~ 1,
                . == "Unchecked" ~ 0,)))

#amend numeric features to numeric format
cl2$weight <- round(as.numeric(cl2$Weight..lbs.), digits =1)
cl2$pack_yrs <- round(as.numeric(cl2$Pack.Years), digits =0)
cl2$quit_smoke_yr <- round(as.numeric(cl2$Quit.Smoking.Year), digits =0)

#keep variables interested in
cl2b <- cl2 |>
  select(-c(Patient.affiliation, 
            Gender, Histology, 
            Ethnicity, 
            X.GG, 
            Pathological.T.stage, 
            Pathological.N.stage, 
            Pathological.M.stage,
            Histopathological.Grade,
            Lymphovascular.invasion,
            EGFR.mutation.status,
            KRAS.mutation.status,
            ALK.translocation.status,
            Weight..lbs.,
            Smoking.status, 
            Pack.Years, 
            Quit.Smoking.Year,
            Recurrence.Location))

#rename long feature names
cl2b <- cl2b |>
  rename(ID = Case.ID,
         Age = Age.at.Histological.Diagnosis,
         tl_RUL = Tumor.Location..choice.RUL.,
         tl_RML = Tumor.Location..choice.RML.,
         tl_RLL = Tumor.Location..choice.RLL.,
         tl_LUL = Tumor.Location..choice.LUL.,
         tl_LLL = Tumor.Location..choice.LLL.,
         tl_LLing = Tumor.Location..choice.L.Lingula.,
         tl_U = Tumor.Location..choice.Unknown.,
         adj_trt = Adjuvant.Treatment)

#amend date features to date format
cl2b <- cl2b |>
  mutate(across(c(CT.Date, Date.of.Death, Date.of.Last.Known.Alive, Date.of.Recurrence,PET.Date), mdy))

#create censoring variables using CT.date
cl2c <- cl2b |>
  mutate(
    end_date = if_else(
      !is.na(Date.of.Death),
      Date.of.Death,
      Date.of.Last.Known.Alive),
    surv = as.numeric(end_date - CT.Date),
    cens = if_else(!is.na(Date.of.Death), 1, 0)) |>
  relocate(CT.Date, Date.of.Death, Date.of.Last.Known.Alive, end_date, surv, cens, Survival.Status)


#create binary survival status from a chosen fixed time point - 3 years
cl2c <- cl2c |>
  mutate(
    surv_3y = if_else(surv > 1095, 1095, surv),
    cens_3y = case_when(
      surv > 1095 & cens == 1 ~ 0,
      TRUE ~ cens),
    status_3y = case_when(
      cens_3y == 1 ~ 1,
      cens_3y == 0 & surv_3y == 1095 ~ 0,
      cens_3y == 0 & surv_3y < 1095 ~ NA_real_))|>
  relocate(ID,status_3y,surv_3y,cens_3y)

#remove vars similar to target to omit data leakage
cl2d <- cl2c |>
  select(-c(Date.of.Death,Date.of.Last.Known.Alive,CT.Date,end_date,Survival.Status,Time.to.Death..days.,Days.between.CT.and.surgery))

#check distribution of continuous vars
#ggplot(cl2d, aes(x = Age)) + geom_histogram(bins = 30) #reasonable
#ggplot(cl2d, aes(x = Age)) + geom_density() #reasonable

#ggplot(cl2d, aes(x = pack_yrs)) + geom_histogram(bins = 30) #reasonable
#ggplot(cl2d, aes(x = pack_yrs)) + geom_density() #reasonable


#Standardise continous data
cl2e <- cl2d
cl2e$age_z <- as.numeric(scale(cl2d$Age))
cl2e$weight_z <- as.numeric(scale(cl2d$weight))
cl2e$pack_yrs_z <- as.numeric(scale(cl2d$pack_yrs))

#output clean clinical data (no missing imputation conducted)
write.csv(cl1c, "Data/Derived/clinical1_clean.csv")
write.csv(cl2e, "Data/Derived/clinical2_clean.csv")


######## Clean RNA Genetic data ##########

#keep only complete data only
rna_cc <- rna |>
  filter(if_all(everything(), ~ !is.na(.)))

#transpose RNA data
rna2 <- as.data.frame(t(rna_cc))
colnames(rna2) <- rna2[1, ]
rna2 <- rna2[-1, ]
rna2$ID <- rownames(rna2)
rna2 <- rna2[, c("ID", setdiff(names(rna2), "ID"))]
rownames(rna2) <- NULL

#remove fully NA features
rna2 <- rna2[, colSums(!is.na(rna2)) > 0] #5269 features remaining

#remove fully NA pts
rna2 <- rna2[rowSums(!is.na(rna2)) > 0, ] #none

#remove fully 0 features or pts
rna2 <- rna2[, colSums(rna2 != 0, na.rm = TRUE) > 0]
rna2 <- rna2[rowSums(rna2 != 0, na.rm = TRUE) > 0, ]

#normalise and standardise RNA data
rna2[-1] <- lapply(rna2[-1], as.numeric)
rna3 <- log2(rna2[, -1] + 1)
rna4 <- as.data.frame(scale(rna3))
rna4$ID <- rna2$ID
rna4 <- rna4[, c("ID", setdiff(names(rna4), "ID"))]

#check standardisation worked
#mean(rna4[, 2], na.rm = TRUE)
#sd(rna4[, 2], na.rm = TRUE)

#amend ID to match the clinical IDS
rna4$ID <- gsub("\\.", "-", rna4$ID)

#Output RNA data
write.csv(rna4, "Data/Derived/RNA_clean.csv")

#merge RNA and clinical2_clean data by ID to combine the data
cl2e_rna4 <- merge(cl2e, rna4, by = "ID", all.x = TRUE)

#output combined dataset
write.csv(cl2e_rna4, "Data/Derived/clinical1_RNA_clean.csv")


#merge RNA (non scaled) and clinical2_clean data by ID to combine the data for missing imputation
#amend ID to match the clinical IDS
rna2$ID <- gsub("\\.", "-", rna2$ID)
cl2d_rna2 <- merge(cl2d, rna2, by = "ID", all.x = TRUE)

######### Dealing with missingness ############

#Assess percentage of missingess for each variable

#use cl2d and rna2 so that we can normalise and scale after imputation

#Clinical1
cl1b_na_summary <- data.frame(
  variable = names(cl1b),
  na_percent = colMeans(is.na(cl1b)) * 100)
#no missingness over 70%

#Clinical2
cl2d_na_summary <- data.frame(
  variable = names(cl2d),
  na_percent = colMeans(is.na(cl2d)) * 100)
#missingness over 70% = Date.of.Recurrence, recur_loc_num


#clinical2 and RNA combined
cl2d_rna2_na_summary <- data.frame(
  variable = names(cl2d_rna2),
  na_percent = colMeans(is.na(cl2d_rna2)) * 100)
cl2d_rna2_na_70 <- cl2d_rna2_na_summary |>
  filter(na_percent>70)
#missingness over 70% = Date.of.Recurrence, recur_loc_num

#Create datasets which have high missingness variables removed

#clinical1 use cl1b - no variables to remove
#clinical2
cl2f <- cl2d |> select(-c(Date.of.Recurrence, recur_loc_num))
#clinical and rna combined
cl2_rna_2 <- cl2d_rna2 |> select(-c(Date.of.Recurrence, recur_loc_num))


########### Split data into Train and Test ###########

#need to split all data (all splits need to be the same), stratified by the target:
#derived - no missingness approaches:          cl1c, cl2e, cl2e_rna4, 
#derived - remove 70% missingness features:    cl1b, cl2f, cl2_rna_2

#data which is censored before 3 years will have NA for status_3y, and so will be randomly assigned into the split data

###################################################
# cl1c - clinical 1 no missing approaches applied # 
###################################################

# Split data into two groups
cl1_trgt_knwn <- cl1c |> filter(status_3y == 0 | status_3y == 1)
cl1_trgt_unknwn <- cl1c |> filter(is.na(status_3y))

# Stratified split for known cases (80/20)
set.seed(123)
cl1_knwn_indices <- createDataPartition(cl1_trgt_knwn$status_3y, p = 0.8, list = FALSE)
cl1_train_knwn <- cl1_trgt_knwn[cl1_knwn_indices, ]
cl1_test_knwn <- cl1_trgt_knwn[-cl1_knwn_indices, ]

# Random split for censored cases (80/20)
set.seed(123)  #same seed for reproducibility
cl1_unknwn_indices <- sample(seq_len(nrow(cl1_trgt_unknwn)), size = 0.8 * nrow(cl1_trgt_unknwn))
cl1_train_unknwn <- cl1_trgt_unknwn[cl1_unknwn_indices, ]
cl1_test_unknwn <- cl1_trgt_unknwn[-cl1_unknwn_indices, ]

# Combine back together
cl1_train <- bind_rows(cl1_train_knwn, cl1_train_unknwn)
cl1_test <- bind_rows(cl1_test_knwn, cl1_test_unknwn)

#Verify proportions
cat("Train set size:", nrow(cl1_train), "\n")
cat("Test set size:", nrow(cl1_test), "\n")
cat("Proportion of events in train:", 
    mean(cl1_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl1_test$status_3y == 1, na.rm = TRUE), "\n")
#happy

###################################################
# cl2e - clinical 2 no missing approaches applied # 
###################################################

# Split data into two groups
cl2_trgt_knwn <- cl2e |> filter(status_3y == 0 | status_3y == 1)
cl2_trgt_unknwn <- cl2e |> filter(is.na(status_3y))

# Stratified split for known cases (80/20)
set.seed(123)
cl2_knwn_indices <- createDataPartition(cl2_trgt_knwn$status_3y, p = 0.8, list = FALSE)
cl2_train_knwn <- cl2_trgt_knwn[cl2_knwn_indices, ]
cl2_test_knwn <- cl2_trgt_knwn[-cl2_knwn_indices, ]

# Random split for censored cases (80/20)
set.seed(123)  #same seed for reproducibility
cl2_unknwn_indices <- sample(seq_len(nrow(cl2_trgt_unknwn)), size = 0.8 * nrow(cl2_trgt_unknwn))
cl2_train_unknwn <- cl2_trgt_unknwn[cl2_unknwn_indices, ]
cl2_test_unknwn <- cl2_trgt_unknwn[-cl2_unknwn_indices, ]

# Combine back together
cl2_train <- bind_rows(cl2_train_knwn, cl2_train_unknwn)
cl2_test <- bind_rows(cl2_test_knwn, cl2_test_unknwn)

#Verify proportions
cat("Train set size:", nrow(cl2_train), "\n")
cat("Test set size:", nrow(cl2_test), "\n")
cat("Proportion of events in train:", 
    mean(cl2_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl2_test$status_3y == 1, na.rm = TRUE), "\n")
#happy

########################################################################
# cl2e_rna4 - clinical2 and genetic data no missing approaches applied # 
########################################################################

# Split data into two groups
cl_rna_trgt_knwn <- cl2e_rna4 |> filter(status_3y == 0 | status_3y == 1)
cl_rna_trgt_unknwn <- cl2e_rna4 |> filter(is.na(status_3y))

# Stratified split for known cases (80/20)
set.seed(123)
cl_rna_knwn_indices <- createDataPartition(cl_rna_trgt_knwn$status_3y, p = 0.8, list = FALSE)
cl_rna_train_knwn <- cl_rna_trgt_knwn[cl_rna_knwn_indices, ]
cl_rna_test_knwn <- cl_rna_trgt_knwn[-cl_rna_knwn_indices, ]

# Random split for censored cases (80/20)
set.seed(123)  #same seed for reproducibility
cl_rna_unknwn_indices <- sample(seq_len(nrow(cl_rna_trgt_unknwn)), size = 0.8 * nrow(cl_rna_trgt_unknwn))
cl_rna_train_unknwn <- cl_rna_trgt_unknwn[cl_rna_unknwn_indices, ]
cl_rna_test_unknwn <- cl_rna_trgt_unknwn[-cl_rna_unknwn_indices, ]

# Combine back together
cl_rna_train <- bind_rows(cl_rna_train_knwn, cl_rna_train_unknwn)
cl_rna_test <- bind_rows(cl_rna_test_knwn, cl_rna_test_unknwn)

#Verify proportions
cat("Train set size:", nrow(cl_rna_train), "\n")
cat("Test set size:", nrow(cl_rna_test), "\n")
cat("Proportion of events in train:", 
    mean(cl_rna_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl_rna_test$status_3y == 1, na.rm = TRUE), "\n")
#happy


#####################################################
# cl1b - clinical 1 WITH missing approaches applied # 
#####################################################

# Split data into two groups
cl1_m_trgt_knwn <- cl1b |> filter(status_3y == 0 | status_3y == 1)
cl1_m_trgt_unknwn <- cl1b |> filter(is.na(status_3y))

# Stratified split for known cases (80/20)
set.seed(123)
cl1_m_knwn_indices <- createDataPartition(cl1_m_trgt_knwn$status_3y, p = 0.8, list = FALSE)
cl1_m_train_knwn <- cl1_m_trgt_knwn[cl1_m_knwn_indices, ]
cl1_m_test_knwn <- cl1_m_trgt_knwn[-cl1_m_knwn_indices, ]

# Random split for censored cases (80/20)
set.seed(123)  #same seed for reproducibility
cl1_m_unknwn_indices <- sample(seq_len(nrow(cl1_m_trgt_unknwn)), size = 0.8 * nrow(cl1_m_trgt_unknwn))
cl1_m_train_unknwn <- cl1_m_trgt_unknwn[cl1_m_unknwn_indices, ]
cl1_m_test_unknwn <- cl1_m_trgt_unknwn[-cl1_m_unknwn_indices, ]

# Combine back together
cl1_m_train <- bind_rows(cl1_m_train_knwn, cl1_m_train_unknwn)
cl1_m_test <- bind_rows(cl1_m_test_knwn, cl1_m_test_unknwn)

#Verify proportions
cat("Train set size:", nrow(cl1_m_train), "\n")
cat("Test set size:", nrow(cl1_m_test), "\n")
cat("Proportion of events in train:", 
    mean(cl1_m_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl1_m_test$status_3y == 1, na.rm = TRUE), "\n")
#happy

#####################################################
# cl2f - clinical 2 WITH missing approaches applied # 
#####################################################

# Split data into two groups
cl2_m_trgt_knwn <- cl2f |> filter(status_3y == 0 | status_3y == 1)
cl2_m_trgt_unknwn <- cl2f |> filter(is.na(status_3y))

# Stratified split for known cases (80/20)
set.seed(123)
cl2_m_knwn_indices <- createDataPartition(cl2_m_trgt_knwn$status_3y, p = 0.8, list = FALSE)
cl2_m_train_knwn <- cl2_m_trgt_knwn[cl2_m_knwn_indices, ]
cl2_m_test_knwn <- cl2_m_trgt_knwn[-cl2_m_knwn_indices, ]

# Random split for censored cases (80/20)
set.seed(123)  #same seed for reproducibility
cl2_m_unknwn_indices <- sample(seq_len(nrow(cl2_m_trgt_unknwn)), size = 0.8 * nrow(cl2_m_trgt_unknwn))
cl2_m_train_unknwn <- cl2_m_trgt_unknwn[cl2_m_unknwn_indices, ]
cl2_m_test_unknwn <- cl2_m_trgt_unknwn[-cl2_m_unknwn_indices, ]

# Combine back together
cl2_m_train <- bind_rows(cl2_m_train_knwn, cl2_m_train_unknwn)
cl2_m_test <- bind_rows(cl2_m_test_knwn, cl2_m_test_unknwn)

#Verify proportions
cat("Train set size:", nrow(cl2_m_train), "\n")
cat("Test set size:", nrow(cl2_m_test), "\n")
cat("Proportion of events in train:", 
    mean(cl2_m_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl2_m_test$status_3y == 1, na.rm = TRUE), "\n")
#happy

##########################################################################
# cl2_rna_2 - clinical2 and genetic data WITH missing approaches applied # 
##########################################################################

# Split data into two groups
cl_rna2_trgt_knwn <- cl2_rna_2 |> filter(status_3y == 0 | status_3y == 1)
cl_rna2_trgt_unknwn <- cl2_rna_2 |> filter(is.na(status_3y))

# Stratified split for known cases (80/20)
set.seed(123)
cl_rna2_knwn_indices <- createDataPartition(cl_rna2_trgt_knwn$status_3y, p = 0.8, list = FALSE)
cl_rna2_train_knwn <- cl_rna2_trgt_knwn[cl_rna2_knwn_indices, ]
cl_rna2_test_knwn <- cl_rna2_trgt_knwn[-cl_rna2_knwn_indices, ]

# Random split for censored cases (80/20)
set.seed(123)  #same seed for reproducibility
cl_rna2_unknwn_indices <- sample(seq_len(nrow(cl_rna2_trgt_unknwn)), size = 0.8 * nrow(cl_rna2_trgt_unknwn))
cl_rna2_train_unknwn <- cl_rna2_trgt_unknwn[cl_rna2_unknwn_indices, ]
cl_rna2_test_unknwn <- cl_rna2_trgt_unknwn[-cl_rna2_unknwn_indices, ]

# Combine back together
cl_rna_m_train <- bind_rows(cl_rna2_train_knwn, cl_rna2_train_unknwn)
cl_rna_m_test <- bind_rows(cl_rna2_test_knwn, cl_rna2_test_unknwn)

#Verify proportions
cat("Train set size:", nrow(cl_rna_m_train), "\n")
cat("Test set size:", nrow(cl_rna_m_test), "\n")
cat("Proportion of events in train:", 
    mean(cl_rna_m_train$status_3y == 1, na.rm = TRUE), "\n")
cat("Proportion of events in test:", 
    mean(cl_rna_m_test$status_3y == 1, na.rm = TRUE), "\n")
#happy

#save split datasets
#standardised datasets to use for classic machine learning
write.csv(cl1_train, "Data/Derived/cl1_train.csv")
write.csv(cl1_test, "Data/Derived/cl1_test.csv")
write.csv(cl2_train, "Data/Derived/cl2_train.csv")
write.csv(cl2_test, "Data/Derived/cl2_test.csv")
write.csv(cl_rna_train, "Data/Derived/cl2_rna_train.csv")
write.csv(cl_rna_test, "Data/Derived/cl2_rna_test.csv")

#non standardised, >70% missingness remove datasets used for neural networks
write.csv(cl1_m_train, "Data/Derived/cl1_m_train.csv")
write.csv(cl1_m_test, "Data/Derived/cl1_m_test.csv")
write.csv(cl2_m_train, "Data/Derived/cl2_m_train.csv")
write.csv(cl2_m_test, "Data/Derived/cl2_m_test.csv")
write.csv(cl_rna_m_train, "Data/Derived/cl2_rna_m_train.csv")
write.csv(cl_rna_m_test, "Data/Derived/cl2_rna_m_test.csv")




####Now, apply K-NN (k=3) to the missingness approach datasets: cl1_m, cl2_m, cl2_rna_m


#Convert Date columns to numeric (days since origin)
convert_dates_to_numeric <- function(df) {
  df[] <- lapply(df, function(col) {
    if (inherits(col, "Date")) {
      as.numeric(col)  #Converts to days since 1970-01-01 (standard R origin)
    } else {
      col
    }
  })
  return(df)
}

# Step 2: Apply conversion to your datasets
cl2_m_train <- convert_dates_to_numeric(cl2_m_train)
cl2_m_test <- convert_dates_to_numeric(cl2_m_test)
cl_rna_m_train <- convert_dates_to_numeric(cl_rna_m_train)
cl_rna_m_test <- convert_dates_to_numeric(cl_rna_m_test)


#Impute with kNN (fits on training set, then is applied to both train and test)
#these code also centers and scales all data at the same time.

set.seed(123)

#Impute clinical 1 data
cl1_train_imputed_pre <- kNN(cl1_m_train, k = 3)
cl1_test_imputed_pre <- kNN(cl1_m_test, k = 3)
#Extract only the original columns
cl1_train_imputed <- cl1_train_imputed_pre[, colnames(cl1_m_train)]
cl1_test_imputed <- cl1_test_imputed_pre[, colnames(cl1_m_test)]

#Impute clinical 2 data
cl2_train_imputed_pre <- kNN(cl2_m_train, k = 3)
cl2_test_imputed_pre <- kNN(cl2_m_test, k = 3)
#Extract only the original columns
cl2_train_imputed <- cl2_train_imputed_pre[, colnames(cl2_m_train)]
cl2_test_imputed <- cl2_test_imputed_pre[, colnames(cl2_m_test)]

#Impute clinical 2 anf RNA merged data - not yet run
cl_rna_train_imputed_pre <- kNN(cl_rna_m_train, k = 3)
cl_rna_test_imputed_pre <- kNN(cl_rna_m_test, k = 3)
#Extract only the original columns
cl_rna_train_imputed <- cl_rna_train_imputed_pre[, colnames(cl_rna_m_train)]
cl_rna_test_imputed <- cl_rna_test_imputed_pre[, colnames(cl_rna_m_test)]

#save imputed datasets
write.csv(cl1_train_imputed, "Data/Derived/cl1_m_train_imputed.csv")
write.csv(cl1_test_imputed, "Data/Derived/cl1_m_test_imputed.csv")
write.csv(cl2_train_imputed, "Data/Derived/cl2_m_train_imputed.csv")
write.csv(cl2_test_imputed, "Data/Derived/cl2_m_test_imputed.csv")

#not yet run
write.csv(cl_rna_train_imputed, "Data/Derived/cl2_rna_m_train_imputed.csv")
write.csv(cl_rna_test_imputed, "Data/Derived/cl2_rna_m_test_imputed.csv")

