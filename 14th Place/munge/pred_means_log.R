preds_mean_log_subset <- train %>% 
  distinct(process_id, object_id) %>% 
  filter(process_id%%10<=7) %>% 
  left_join(train_labels) %>% 
  rename(target = final_rinse_total_turbidity_liter) %>% 
  filter(target<7500000) %>% 
  group_by(object_id) %>% 
  summarise(
    median_target = median(log10(target))
  )

preds_mean__log_all <- train %>% 
  distinct(process_id, object_id) %>% 
  left_join(train_labels) %>% 
  rename(target = final_rinse_total_turbidity_liter) %>% 
  filter(target<7500000) %>% 
  group_by(object_id) %>% 
  summarise(
    median_target = median(log10(target))
  )