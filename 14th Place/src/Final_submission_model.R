# Function that summarises the data ---------------------------------------

convert_summary <- function(train, PHASES){
  train %>% 
    filter(phase%in%PHASES) %>%
    group_by(process_id, object_id) %>% 
    summarise(
      the_target = median(supply_flow*return_turbidity),
      the_target_sd = sd(supply_flow*return_turbidity),
      return_turbidity = mean(return_turbidity),
      return_temperature = mean(return_temperature),
      supply_flow_mean = mean(supply_flow),
      supply_flow_median = median(supply_flow),
      pipeline = unique(pipeline),
      supply_flow_diff = median(supply_flow)-mean(supply_flow),
      return_conductivity = mean(return_conductivity),
      supply_pressure =mean(supply_pressure),
      tank_level_pre_rinse = mean(tank_level_pre_rinse),
      supply_pump  =mean(supply_pump ==  'True'),
      tank_concentration_caustic =mean(tank_concentration_caustic),
      object_low_level = mean(object_low_level=='True'),
      tank_level_clean_water = mean( tank_level_clean_water),
      return_flow1  = mean(return_flow ),
      return_drain = mean(return_drain=='True'),
      supply_pre_rinse1 = mean(supply_pre_rinse =='True')
    ) %>% ungroup() %>% 
    left_join(preds_mean_all, by = 'object_id') %>% 
    left_join(Recipe,  by = "process_id") 
} 

convert_train_summary <- function(data,...){
  convert_summary(data,...) %>% 
    left_join(train_labels, by = "process_id") %>% 
    rename(target = final_rinse_total_turbidity_liter) %>%  
    filter(target< 7500000) 
}

# Summarise data to train models ------------------------------------------

#0001
data_acid <- convert_train_summary(train, 'acid') %>% 
  dplyr::select(the_target:supply_pump, return_flow1,
                tank_concentration_caustic:tank_level_clean_water,
                target, median_target)

#0100
data_caustic <- convert_train_summary(train, 'caustic') %>% 
  filter(pipeline=='L3') %>% 
  dplyr::select(the_target:tank_concentration_caustic, return_drain,
                object_low_level, tank_level_clean_water,
                recipe, target, median_target, -supply_flow_median)

#11100
data_first3_phases <- convert_train_summary(train, 'intermediate_rinse') %>% 
  dplyr::select(the_target:supply_pump, return_flow1,
                tank_concentration_caustic:tank_level_clean_water,
                recipe, target, median_target)

#Only recipes with 11111 and acid phase recorded
data_all_phases <- convert_train_summary(train, 'acid') %>% 
  dplyr::select(the_target:supply_pump, return_flow1, supply_pre_rinse1, object_low_level,
                tank_level_clean_water, recipe, target, median_target)

#Only saw pre-rinse 1000
data_prerinse <- convert_train_summary(train, 'pre_rinse') %>% 
  dplyr::select(the_target:supply_pump, return_flow1, supply_pre_rinse1,
                object_low_level, tank_level_clean_water, recipe, target, median_target)

#Pre-rinse and caustic 1100
data_first2_phases <- convert_train_summary(train, c('caustic', 'pre_rinse')) %>% 
  dplyr::select(the_target:supply_pump, return_flow1, supply_pre_rinse1,
                object_low_level, tank_level_clean_water, recipe, target, median_target)

# Train models ------------------------------------------------------------
library(ranger)

train_model <- function(the_data){
  ranger(formula = log10(target)~., data = the_data, 
         seed = 100243, importance = 'impurity',num.trees = 700, mtry = 6)
}

rf_acid <- train_model(data_acid)
rf_caustic <- train_model(data_caustic)
rf_first3_phases <- train_model(data_first3_phases)
rf_all_phases <- train_model(data_all_phases)
rf_prerinse <- train_model(data_prerinse)
rf_first2_phases <- train_model(data_first2_phases)

# Prepare data to predict -------------------------------------------------

phases_in_process <-   test %>% group_by(process_id, pipeline) %>% 
  summarise(
    pre_rinse = 'pre_rinse'%in%unique(phase),
    caustic = 'caustic'%in%unique(phase),
    int_rinse = 'intermediate_rinse'%in%unique(phase),
    acid = 'acid'%in%unique(phase)
  ) %>% left_join(Recipe)

pipeline_mult <- function(test_pred, the_coef){
  
  pipe_mult <- data.frame(pipeline = paste0('L', c(1:4,6:11)),
                          coeff = the_coef)
  
  test_pred %<>% left_join(pipe_mult, by = 'pipeline') %>% 
    mutate(new_preds = preds * coeff)
  
}

#0001
subset_0001 <- phases_in_process %>% left_join(Recipe) %>%  
  filter( acid == T, recipe!=11111) %$% 
  process_id 

test_acid <- convert_summary(test, 'acid') %>% 
  filter(process_id%in% subset_0001) %>% 
  mutate(preds = 10^(predict(rf_acid, .)$predictions)) %>% 
  pipeline_mult(the_coef = c(.93,.93,.85,.95,.93,.85,.93,.85,1,.93))

#0100
subset_0100 <- phases_in_process %>% 
  filter(pre_rinse==F, caustic ==T, int_rinse==F, acid == F, pipeline =='L3') %$% 
  process_id 

test_caustic <- convert_summary(test, 'caustic') %>% 
  filter(process_id%in% subset_0100) %>% 
  mutate(preds = 10^(predict(rf_caustic, .)$predictions)) %>% 
  pipeline_mult(the_coef = c(.85))

#1110
subset_1110 <- phases_in_process %>% filter(int_rinse==T, acid == F) %$%
  process_id

test_first3_phases <- convert_summary(test, 'intermediate_rinse') %>% 
  filter(process_id%in% subset_1110) %>% 
  mutate(preds = 10^(predict(rf_first3_phases, .)$predictions)) %>% 
  pipeline_mult(the_coef = c(.93,.93,.84,.95,.93,.84,.93,.73,.95,.93))

#Only recipes with 1111 and acid phase recorded
subset_1111 <- phases_in_process_test %>% 
  filter(recipe==11111,acid == T) %$%
  process_id 

test_all_phases <- convert_summary(test, 'acid') %>% 
  filter(process_id%in% subset_1111) %>% 
  mutate(preds = 10^(predict(rf_all_phases, .)$predictions)) %>% 
  pipeline_mult(the_coef = c(.97,.97,.82,.97,.97,.82,.97,.85,1,.97))

#Only saw pre-rinse 1000
subset_1000 <- phases_in_process %>% 
  filter(pre_rinse==T, caustic ==F, int_rinse==F, acid == F) %$%
  process_id 

test_prerinse <- convert_summary(test, 'pre_rinse')  %>% 
  filter(process_id%in% subset_1000) %>% 
  mutate(preds = 10^(predict(rf_prerinse, .)$predictions)) %>% 
  pipeline_mult(the_coef = c(.87,.87,.83,.94,.87,.83,.87,.64,.94,.87))

#Pre-rinse and caustic 1100
subset_1100 <- phases_in_process_test %>%
  filter(pre_rinse==T, caustic ==T, int_rinse==F, acid == F) %$%   
  process_id 

test_first2_phases <- convert_summary(test, c('caustic', 'pre_rinse')) %>% 
  filter(process_id%in% subset_1100) %>% 
  mutate(preds = 10^(predict(rf_first2_phases, .)$predictions)) %>% 
  pipeline_mult(the_coef = c(.93,.93,.83,.95,.93,.83,.93,.73,.95,.93))

# Join all the predictions ------------------------------------------------

Submission <- bind_rows(test_acid, test_caustic, test_first3_phases, test_first2_phases,
                        test_all_phases, test_prerinse) %>% 
  select(process_id, new_preds) %>% 
  rename(final_rinse_total_turbidity_liter = new_preds) %>% 
  arrange(process_id)



