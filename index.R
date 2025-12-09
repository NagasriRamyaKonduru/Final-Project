############################################################
# Big Data Econometrics Final Project
# Predicting Remaining Subscription Months for a Streaming Platform
# Author: Nagasri Ramya Konduru
# Course: ECG 564 Big Data Econometrics
############################################################

# ---- 0. Setup ----

set.seed(12345)

# Install packages once (if needed):
# install.packages(c("tidyverse", "caret", "glmnet", "rpart",
#                    "rpart.plot", "randomForest", "xgboost"))

library(tidyverse)
library(caret)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)

# ---- 1. Load Data ----
# TODO: change path/filename to your actual Kaggle CSV
data_raw <- data

# Quick check
glimpse(data_raw)

# ---- 2. Variable Construction ----
# IMPORTANT:
# If your dataset already has the remaining months variable, rename it to `remaining_months`.
# If you only have, say, churn date vs. observation date, you’ll compute the difference in months,
# then cap at 12. Example (commented out):

# data_raw <- data_raw %>%
#   mutate(
#     remaining_months = pmin(12,
#       as.numeric(difftime(subscription_end_date, observation_date, units = "days")) / 30
#     )
#   )

# For now, we assume `remaining_months` already exists and is numeric 0–12.

# ---- 2a. Choose variables consistent with proposal ----
# Adjust the names here to match your actual dataset’s column names
data <- data_raw %>%
  transmute(
    remaining_months,                 # outcome Y
    account_age_months,              # account age
    plan_type,                       # Basic/Standard/Premium
    hours_watched_last_month,
    active_days_last_month,
    logins_per_week,
    share_mobile,                    # fraction of time on mobile
    share_tv,                        # fraction of time on TV
    share_laptop,                    # fraction of time on laptop
    genre_diversity,                 # genre diversity index
    avg_binge_length,                # average episodes per session
    payment_failures_6m,
    paused,                          # 0/1
    shared_account                   # 0/1
  )

# Make sure types are correct
data <- data %>%
  mutate(
    plan_type = factor(plan_type),
    paused = factor(paused, levels = c(0, 1)),
    shared_account = factor(shared_account, levels = c(0, 1))
  )

# Drop any rows with missing values (you can instead impute if needed)
data <- na.omit(data)

# Quick summary
summary(data)

# ---- 3. Train–Test Split ----
set.seed(12345)
train_index <- createDataPartition(data$remaining_months, p = 0.7, list = FALSE)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# ---- 4. Helper: Performance Metrics ----

metrics <- function(y_true, y_pred) {
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae  <- mean(abs(y_true - y_pred))
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2   <- 1 - ss_res / ss_tot
  tibble(RMSE = rmse, MAE = mae, R2 = r2)
}

results_list <- list()  # to store metrics

# ---- 5. OLS Regression ----

ols_formula <- remaining_months ~ .

ols_model <- lm(ols_formula, data = train_data)
summary(ols_model)

ols_pred <- predict(ols_model, newdata = test_data)
results_list$OLS <- metrics(test_data$remaining_months, ols_pred)

# ---- 6. Ridge and LASSO (glmnet) ----

# Create model matrices (dummy variables automatically created)
x_train <- model.matrix(remaining_months ~ ., data = train_data)[, -1]
y_train <- train_data$remaining_months

x_test  <- model.matrix(remaining_months ~ ., data = test_data)[, -1]
y_test  <- test_data$remaining_months

set.seed(12345)
# Ridge: alpha = 0
cv_ridge <- cv.glmnet(
  x_train, y_train,
  alpha = 0,
  nfolds = 10,
  standardize = TRUE
)

ridge_lambda <- cv_ridge$lambda.min
ridge_lambda

ridge_model <- glmnet(
  x_train, y_train,
  alpha = 0,
  lambda = ridge_lambda,
  standardize = TRUE
)

ridge_pred <- predict(ridge_model, newx = x_test)
results_list$Ridge <- metrics(y_test, ridge_pred)

# LASSO: alpha = 1
set.seed(12345)
cv_lasso <- cv.glmnet(
  x_train, y_train,
  alpha = 1,
  nfolds = 10,
  standardize = TRUE
)

lasso_lambda <- cv_lasso$lambda.min
lasso_lambda

lasso_model <- glmnet(
  x_train, y_train,
  alpha = 1,
  lambda = lasso_lambda,
  standardize = TRUE
)

lasso_pred <- predict(lasso_model, newx = x_test)
results_list$LASSO <- metrics(y_test, lasso_pred)

# Inspect nonzero coefficients for LASSO
lasso_coefs <- coef(lasso_model)
lasso_coefs

# ---- 7. Regression Tree (CART) ----

set.seed(12345)
tree_model <- rpart(
  remaining_months ~ .,
  data = train_data,
  method = "anova",
  control = rpart.control(cp = 0.001, minsplit = 20)
)

printcp(tree_model) # see cp table to pick the best cp
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]

tree_pruned <- prune(tree_model, cp = best_cp)

rpart.plot(tree_pruned, main = "Regression Tree for Remaining Months")

tree_pred <- predict(tree_pruned, newdata = test_data)
results_list$Tree <- metrics(test_data$remaining_months, tree_pred)

# ---- 8. Random Forest ----

set.seed(12345)
rf_model <- randomForest(
  remaining_months ~ .,
  data = train_data,
  ntree = 500,
  mtry  = floor(sqrt(ncol(train_data) - 1)),  # rule-of-thumb
  importance = TRUE
)

rf_model

rf_pred <- predict(rf_model, newdata = test_data)
results_list$RandomForest <- metrics(test_data$remaining_months, rf_pred)

# Variable importance plot
varImpPlot(rf_model, main = "Random Forest Variable Importance")

# ---- 9. Gradient Boosting (XGBoost) ----
# Use the same x_train/x_test matrices

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

# Basic parameter grid (you can tweak if time allows)
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(12345)
xgb_model <- xgb.train(
  params = params,
  data   = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain),
  verbose = 0
)

xgb_pred <- predict(xgb_model, newdata = dtest)
results_list$XGBoost <- metrics(y_test, xgb_pred)

# ---- 10. Combine and Export Results --z--

model_comparison <- bind_rows(results_list, .id = "Model") %>%
  arrange(RMSE)

print(model_comparison)

# Save to CSV for easy use in Overleaf / Word
write_csv(model_comparison, "model_comparison_results.csv")

# Also save key tuning outputs for your report
tuning_summary <- list(
  ridge_lambda = ridge_lambda,
  lasso_lambda = lasso_lambda,
  tree_best_cp = best_cp,
  rf_ntree     = rf_model$ntree,
  rf_mtry      = rf_model$mtry
)

tuning_summary
############################################################
# END OF SCRIPT
############################################################
