#------------------------------------------------------------------------------
#                          Change data names and create useful data objects
#------------------------------------------------------------------------------



# Train and test ----------------------------------------------------------

train <- train.values
train$phase <- factor(train$phase, 
                      levels = c('pre_rinse', 'caustic', 'intermediate_rinse', 'acid', 'final_rinse')) 

train_labels <- train.labels

test <- test.values
test$phase <- factor(test$phase, 
                     levels = c('pre_rinse', 'caustic', 'intermediate_rinse', 'acid')) 


# Submission and recipe ---------------------------------------------------

submit <- submission.format

recipe <- recipe.metadata
  
Recipe <- recipe %>% mutate(
  recipe = paste0(pre_rinse, caustic, intermediate_rinse, acid, final_rinse) %>% as.character
) %>% select(recipe, process_id)

# Create a pipeline object to compare results -----------------------------

pipeline_object <- train %>% select(object_id, pipeline) %>% distinct()