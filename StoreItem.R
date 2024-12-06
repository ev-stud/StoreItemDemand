library(tidymodels)
library(vroom)


trainstore <- vroom("./StoreItemDemand/train.csv")
teststore <- vroom("./StoreItemDemand/test.csv")


# EDA ---------------------------------------------------------------------
library(ggplot2)
library(patchwork)

# start with one combination of store and item
storeItem <- trainstore %>%
  filter(store == 9, item == 47)

# time series plot
series_7_10 <- storeItem %>% ggplot(mapping = aes(x=date, y=sales)) + 
  geom_line() + geom_smooth(se=F)

# autocorrelation function plots - for timeseries
acf_7_10 <- storeItem %>% 
  pull(sales) %>%
  forecast::ggAcf(.) # notice patterns every 7 days, 14 days, 21 ...

acflag_7_10 <- storeItem %>% 
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365) # notice patterns yearly (365 days)


series_9_47 <- storeItem %>% ggplot(mapping = aes(x=date, y=sales)) + 
  geom_line() + geom_smooth(se=F)

acf_9_47 <- storeItem %>% 
  pull(sales) %>%
  forecast::ggAcf(.)

acflag_9_47 <- storeItem %>% 
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365)

(series_7_10 + acf_7_10 + acflag_7_10) / (series_9_47 + acf_9_47 + acflag_9_47)


# Feature Engineering -----------------------------------------------------
my_recipe <- recipe(sales~., storeItem) %>%
  step_date(date, features="dow") %>%
  step_date(date, features="decimal") %>%
  step_date(date, features="month") %>%
  step_range(date_decimal, min=0, max=pi) %>% 
  step_mutate(sinDec=sin(date_decimal), cosDec=cos(date_decimal)) #%>%
  #step_select(., -date, -store, -item)

library(embed)
item_recipe <- recipe(sales~., data=train) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())
  

baked <- bake(prep(my_recipe), storeItem)

baked %>% ggplot(mapping = aes(x=cosDec, y=sales)) + 
  geom_line() + geom_smooth(se=F)


# Machine Learning Models -------------------------------------------------

### SVM
svmLinear <- svm_linear(cost=tune()) %>% # set or tune cost penalty
  set_mode("regression") %>%
  set_engine("kernlab")

svmPoly <- svm_poly(cost=tune()) %>% # set or tune cost penalty
  set_mode("regression") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(svmPoly)

tuning_grid <- grid_regular(cost(),
                            levels = 3) # grid of L^2 tuning possibilities

folds <- vfold_cv(storeItem, v = 5, repeats =1) # K-folds

cv_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

cv_results %>%
  select_best(metric="smape") 

### SARIMA
library(modeltime)
library(timetk)

train <- trainstore %>% filter(store==7, item==10)
test <- teststore %>% filter(store==7, item==10)

cv_split <- time_series_split(train, assess="3 months", cumulative = TRUE)

# visualize our training set
cv_split %>% 
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales~., data=train) %>%
  step_rm(item, store) %>%
  step_date(date, features=c("doy", "decimal")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_rm(date_doy)

# create arima "tuning grid" for parameters
arima_model <- arima_reg(seasonal_period = "3 months", # seasonal period
                         non_seasonal_ar = 7, # tune p up to this max (past time points)
                         seasonal_ar = 2, # tune P up to this max (past season timepoints)
                         non_seasonal_ma = 7, # tune q up to this max (past residuals)
                         seasonal_ma = 2, # tune Q up to this max (past season residuals)
                         non_seasonal_differences = 2, # tune d up to this max (time differences)
                         seasonal_differences = 2 # tune D up to this max (seasonal differences)
                         ) %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split)) # create training set

# calibrate (tune)
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split)) # create testing set

## plot results with specified store/item combo
p3 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )
## Refit to whole data
fullfit <- cv_results %>%
  modeltime_refit(data = train)

# plot predictions with specified store/item combo
p4 <- fullfit %>%
  modeltime_forecast(
    new_data    = test,
    actual_data = train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plotly::subplot(p1,p3,p2,p4, nrows=2)


### Prophet Model
library(modeltime)
library(timetk)

train <- trainstore %>% filter(store==9, item==11)
test <- teststore %>% filter(store==9, item==11)

cv_split <- time_series_split(train, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine="prophet") %>%
  fit(sales ~ date, data = training(cv_split))

# calibrate (tune) the model (not workflow)
cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split)) # create testing set

# visualize and evaluate CV accuracy
p3 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE)

# Refit to whole data
fullfit <- cv_results %>%
  modeltime_refit(data = train)

# plot predictions with specified store/item combo
p4 <- fullfit %>%
  modeltime_forecast(
    new_data    = test,
    actual_data = train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the fullfit accuracy
fullfit %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE)

plotly::subplot(p1,p3,p2,p4, nrows=2)


# Submit to Kaggle --------------------------------------------------------
library(tidymodels)
library(vroom)
library(modeltime)
library(timetk)

train <- vroom("./StoreItemDemand/train.csv")
test <- vroom("./StoreItemDemand/test.csv")

nStores <- max(train$store)
nItems <- max(train$item)

### Prophet model

for(s in 1:nStores){
  for(i in 1:nItems){
    
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>% 
      filter(store==s, item==i)
    
    # Model
    cv_split <- time_series_split(storeItemTrain, 
                                  initial="1 year", # get previous year as training
                                  assess="1 year", # get recent year as testing
                                  cumulative = FALSE) # TRUE ignores 'initial'
    
    prophet_model <- prophet_reg() %>%
      set_engine(engine="prophet") %>%
      fit(sales ~ date, data = training(cv_split))
    
    cv_results <- modeltime_calibrate(prophet_model,
                                      new_data = testing(cv_split))
    
    fullfit <- cv_results %>%
      modeltime_refit(data = storeItemTrain)
    
    # Predictions
    preds <- fullfit %>% 
      modeltime_forecast(new_data = storeItemTest,
                         actual_data = storeItemTrain) %>%
      filter(!is.na(.model_id)) %>%
      mutate(id=storeItemTest$id) %>%
      select(id, .value) %>%
      rename(sales=.value)
    
    # Save StoreItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

vroom_write(all_preds, file="./StoreItemDemand/allPreds", delim=",") # 16.9 score


### SVM

my_recipe <- recipe(sales~., data=train) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())

svmPoly <- svm_poly(cost=0.177) %>% # determined by random store-item combo
  set_mode("regression") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)


for(s in 1:nStores){
  for(i in 1:nItems){
    
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>% 
      filter(store==s, item==i)
    
    
    final_wf <- svm_wf %>%
      finalize_workflow(bestTune) %>% 
      fit(data=storeItemTrain)
    
    # convert to tabletime format
    models_tbl <- modeltime_table(final_wf)
    
    fullfit <- models_tbl %>%
      modeltime_refit(data = storeItemTrain)
    
    # Predictions
    preds <- fullfit %>% 
      modeltime_forecast(new_data = storeItemTest,
                         actual_data = storeItemTrain) %>%
      filter(!is.na(.model_id)) %>%
      mutate(id=storeItemTest$id) %>%
      select(id, .value) %>%
      rename(sales=.value)
    
    # Save StoreItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

