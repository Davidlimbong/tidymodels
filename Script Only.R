#----------------------------- Tidymodels Script---------------------------

# Load Packages -----------------------------------------------------------
library(tidyverse) # data wrangling
library(inspectdf) # data exploration
library(tidymodels) # modeling
library(themis)


# Read Data ---------------------------------------------------------------
churn_df <- read_csv("data/watson-churn.csv")


# Data Preprocessing ------------------------------------------------------

## Cross Validation --------------------------------------------------------
set.seed(123)
churn_split <- initial_split(data = churn_df,prop = 0.8, strata = Churn)
churn_train <- training(churn_split)
churn_test <- testing(churn_split)

## Recipes -----------------------------------------------------------------
churn_rec <- recipe(formula = Churn~., data = churn_train) %>% 
  update_role(customerID, new_role = "ID") %>%
  step_string2factor(all_nominal(), -customerID, skip = T) %>% # akan di skip ketika predict
  step_num2factor(SeniorCitizen, transform = function(x) x +1, levels = c("No", "Yes")) %>%
  step_medianimpute(TotalCharges) %>% 
  step_upsample(Churn,over_ratio = 4/5)


# Modeling --------------------------------------------------------------

## Model Interface (RF)---------------------------------------------------------
rf_model <- rand_forest(mtry = tune(),
                        trees = tune(), 
                        min_n =tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")


## Grid Search (RF) ------------------------------------------------------------
set.seed(123)
rf_grid <- grid_max_entropy(x=finalize(object = mtry(),x = churn_train[,-19]), 
                               trees(), 
                               min_n(), 
                               size = 10)



## Model Interface (XGBoost) ---------------------------------------------------
xgb_model <- boost_tree(mtry = tune(), 
                        trees = tune(),
                        min_n = tune(),
                        tree_depth = tune(), 
                        learn_rate = tune(), 
                        loss_reduction = tune(),
                        sample_size = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")




## Grid Search (XGBoost) -------------------------------------------------------
set.seed(123)
xgb_grid <- grid_max_entropy(x=finalize(object = mtry(),x = churn_train[,-19]), 
                               trees(), 
                               min_n(), 
                               tree_depth(), 
                               learn_rate(), 
                               loss_reduction(),
                             sample_size = sample_prop(),
                               size = 20)



## Metrics Evaluation ------------------------------------------------------
# options(yardstick.event_first = FALSE)
churn_metrics <- metric_set(roc_auc, specificity)

## K-Fold ------------------------------------------------------------------
churn_folds <- vfold_cv(data = churn_train, v = 5)


## Tuning Parameters (RF) ------------------------------------------------------
doParallel::registerDoParallel()
set.seed(123)
rf_tune <- tune_grid(object = rf_model,
                       preprocessor = churn_rec,
                       resamples = churn_folds,
                       grid = rf_grid, 
                       metrics = churn_metrics)

rf_tune %>% 
  collect_metrics() %>% 
  group_by(.metric) %>% 
  slice_max(mean,n = 2) %>% 
  select(.metric, mean)

## Tuning Parameters (XGBoost) ------------------------------------------------------
doParallel::registerDoParallel()
set.seed(123)
xgb_tune <- tune_grid(object = xgb_model,
                      preprocessor = churn_rec %>% 
                         step_dummy(all_nominal(), -customerID, -Churn),
                      resamples = churn_folds,
                      grid = xgb_grid, 
                      metrics = churn_metrics)


xgb_tune %>% 
  collect_metrics() %>% 
  group_by(.metric) %>% 
  slice_max(mean,n = 2) %>% 
  select(.metric, mean)


## Finalization ------------------------------------------------------------
churn_wf <- workflow() %>% 
  add_model(xgb_model) %>% 
  add_recipe(churn_rec %>% 
               step_dummy(all_nominal(), -customerID, -Churn)) %>% 
  finalize_workflow(xgb_tune %>% 
                      show_best("roc_auc", 1)
  )


churn_modelfinal <- fit(object = churn_wf, data = churn_train)


# Model Evaluation --------------------------------------------------------

## Prediction --------------------------------------------------------------
pred_prob <- predict(churn_modelfinal, churn_test, type = "prob")
pred_class <- predict(churn_modelfinal, churn_test, type = "class")

pred_full <- churn_test %>% 
  transmute(truth = as.factor(Churn)) %>% 
  bind_cols(pred_prob, pred_class)

## Confusion Metrics -------------------------------------------------------
pred_full %>% 
  conf_mat(truth, .pred_class) %>% 
  autoplot(type = "heatmap")


# specificity --------------------------------------------------------------
pred_full %>% 
  specificity(truth, .pred_class) 


# ROC Curve ---------------------------------------------------------------
churn_results %>% 
  roc_curve(truth, .pred_Yes,  event_level = 'second') %>% 
  autoplot()


# ROC AUC -----------------------------------------------------------------
pred_full %>% 
  roc_auc(truth, .pred_Yes,  event_level = 'second')






