

# Initialize session ------------------------------------------------------

# Load packages


library(tm)
library(ngram)
library(plyr)
library(dplyr)
library(caret)
library(doMC) # library(parallel); library(doParallel)
library(RSNNS)
library(class)
library(klaR)
library(kernlab)
library(ranger)
library(NLP)
library(stringr)
library(openxlsx)
library(BSDA)

# Get session info and set max.print

sessionInfo()
options(max.print=100000)

# Explore class distribution

barplot(prop.table(table(data$class)), main = "Figure 1: Class Distribution", xlab = "Class", ylab = "Frequency", ylim = c(0,0.5))

# Run models --------------------------------------------------------------

# Pre-process data

data$usable_text <- sapply(data$text, function(x) iconv(enc2utf8(x), sub = "byte"))

pre_process_data <- function(dataset){
  #TODO: Add stop-word remover (make it language specific), stemming
  
  processed_dataset <- VCorpus(VectorSource(dataset$usable_text))
  processed_dataset <- tm_map(processed_dataset, content_transformer(tolower))
  processed_dataset <- tm_map(processed_dataset, removeNumbers)
  processed_dataset <- tm_map(processed_dataset, removePunctuation)
  processed_dataset <- tm_map(processed_dataset, stripWhitespace)
  UnigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 1), paste, collapse = " "), use.names = FALSE)
  dtm_unigram <- DocumentTermMatrix(processed_dataset, control = list(tokenize = UnigramTokenizer, wordLengths=c(3,20), bounds = list(global = c(4,Inf))))
  dtm_unigram <- weightBin(dtm_unigram)
  
  BigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
  dtm_bigram <- DocumentTermMatrix(processed_dataset, control = list(tokenize = BigramTokenizer, wordLengths=c(6,20), bounds = list(global = c(7,Inf))))
  dtm_bigram <- weightBin(dtm_bigram)
  
  return(as.data.frame(cbind(as.matrix(dtm_unigram), as.matrix(dtm_bigram))))
  
}





corpus_clean <- pre_process_data(data)
object.size(corpus_clean)

# Split labeled DTM into 80% test and 20% training set

labeled_dtm <- as.data.frame(cbind(data$class, corpus_clean))
labeled_dtm[,-1] <- apply(labeled_dtm[,-1], 2, function(x) as.numeric(as.character(x)))
colnames(labeled_dtm)[1] <- "class_dv"

set.seed(128); trainIndex <- createDataPartition(data$class, p = .8, list = FALSE)
train_labeled <- labeled_dtm[trainIndex,]
test_labeled <- labeled_dtm[-trainIndex,]

# Create labeled data sets for NB with factor features

train_labeled_nb <- train_labeled
train_labeled_nb[,-1] <- lapply(train_labeled_nb[, -1], factor)
train_labeled_nb[,-1] <- catcolwise(function(v) factor(v, levels = c("0", "1")))(train_labeled_nb[-1])

test_labeled_nb <- test_labeled
test_labeled_nb[,-1] <- lapply(test_labeled_nb[, -1], factor)
test_labeled_nb[,-1] <- catcolwise(function(v) factor(v, levels = c("0", "1")))(test_labeled_nb[-1])

labeled_dtm_nb <- labeled_dtm
labeled_dtm_nb[,-1] <- lapply(labeled_dtm_nb[, -1], factor)
labeled_dtm_nb[,-1] <- catcolwise(function(v) factor(v, levels = c("0", "1")))(labeled_dtm_nb[-1])

# Define trainControl

ctrl_tune <- trainControl(method = "repeatedcv", number = 5, repeats = 1, selectionFunction = "best", 
                          verboseIter = TRUE, savePredictions = "final", classProbs = FALSE)
ctrl_final <- trainControl(method = "repeatedcv", number = 10, repeats = 5, selectionFunction = "best", 
                           verboseIter = TRUE, savePredictions = "final", classProbs = FALSE)

# Set parameter grids

grid_ann <- expand.grid(layer1 = c(1,2,4,8,16,32), layer2 = 0, layer3 = 0, decay = 1e-04)
grid_knn <- expand.grid(k = c(1,15,30,45,65))
grid_nb <- expand.grid(fL = c(0,1), usekernel = FALSE, adjust = 1)
grid_rf <- expand.grid(mtry = c(round(sqrt(ncol(train_labeled))/2),round(sqrt(ncol(train_labeled)))), 
                       splitrule = "gini", min.node.size = 1)
grid_svm <- expand.grid(C = c(0.01,0.1,1,10,100))

# Create set of models and combine grids

set_of_models <- c("mlpWeightDecayML", "knn", "nb", "ranger", "svmLinear")
model_parameter_grids <- as.data.frame(matrix(nrow = length(set_of_models), ncol = 2))

colnames(model_parameter_grids) <- c("model", "parameter_grid")
model_parameter_grids$model = set_of_models
model_parameter_grids$parameter_grid = list(grid_ann, grid_knn, grid_nb, grid_rf, grid_svm)

df_train_results <- as.data.frame(matrix(nrow = length(set_of_models), ncol = 5))
colnames(df_train_results) <- c("final_model", "model", "train_acc", "tuned_parameters", "runtime")

# Initialize lists

models = list()
final_model_list = list()
tuned_parameters = list()
train_predictions <- as.data.frame(matrix(nrow = nrow(train_labeled), ncol = length(set_of_models) + 1))
colnames(train_predictions) <- c("class", "ANN", "kNN", "NB", "RF", "SVM")
train_predictions$class <- train_labeled$class_dv

models_final = list()
final_model_list_final = list()
runtime = list()

# Initialize parallel processing

detectCores() # cluster <- makeCluster(detectCores() - 24)
registerDoMC(cores = 63) # registerDoParallel(cluster)

# Train models

#TODO: Heat-maps for the full confusion matrix to see if certain categories are worse for certain methods - to add features specifically
# for those etc.

#IDEA: Topic modeling LDA: Check if the topics correlate with classes - use the topics as a feature
#IDEA: Models should be evaluated by language e.g. German has longer words vs. english, simple cleaning may leader to higher performance
#IDEA: Word embeddings: an alternative to LDA - can try if there's time
#FEATURES: Other meta features, e.g., days to conduction, parsing text to find the booking_id/date and associated data,
# subject-line (use bag-of-words or something similar)
# count of unigram/bigrams // tf-idf: for each category, you could take the top uni/bi-grams and take the tf-idf values otherwise.


set.seed(128); system.time(
  for(i in 1:length(set_of_models)) {
    
    method_train <- model_parameter_grids$model[i]
    grid <- model_parameter_grids$parameter_grid[i]
    grid <- grid[[1]]
    
    if(method_train != "nb") {
      
      fitted <- caret::train(y = factor(train_labeled[,1]), x = train_labeled[,-1], method = method_train, metric = "Accuracy",
                             tuneGrid = grid, trControl = ctrl_tune)
      
      final_model <- fitted # recommended not to work w/ $finalModel
      train_acc <- caret::confusionMatrix(fitted$pred$pred, fitted$pred$obs)
      
      final_model_list[[i]] <- final_model
      models[[i]] <- fitted
      runtime[[i]] <- fitted$times$final[3]
      tuned_parameters[[i]] <- fitted$bestTune
      
      sorted_predictions <- fitted$pred %>% arrange(rowIndex)
      train_predictions[,i+1] <- as.numeric(as.character(sorted_predictions$pred))
      
      df_train_results$train_acc[i] <- round(train_acc$overall[1],4)
      
      # fit tuned model on full dataset
      
      fitted_final <- caret::train(y = factor(labeled_dtm[,1]), x = labeled_dtm[,-1], method = method_train, metric = "Accuracy",
                                   tuneGrid = fitted$bestTune, trControl = ctrl_final)
      
      final_model_final <- fitted_final # recommended not to work w/ $finalModel
      repeated_acc <- caret::confusionMatrix(fitted_final$pred$pred, fitted_final$pred$obs)
      
      final_model_list_final[[i]] <- final_model_final
      models_final[[i]] <- fitted_final
      
      df_train_results$repeated_acc[i] <- round(repeated_acc$overall[1],4)
      
    }
    else {
      
      fitted <- caret::train(y = factor(train_labeled_nb[,1]), x = train_labeled_nb[,-1], method = method_train, metric = "Accuracy",
                             tuneGrid = grid, trControl = ctrl_tune)
      
      final_model <- fitted # recommended not to work w/ $finalModel
      train_acc <- caret::confusionMatrix(fitted$pred$pred, fitted$pred$obs)
      
      final_model_list[[i]] <- final_model
      models[[i]] <- fitted
      runtime[[i]] <- fitted$times$final[3]
      tuned_parameters[[i]] <- fitted$bestTune
      
      sorted_predictions <- fitted$pred %>% arrange(rowIndex)
      train_predictions[,i+1] <- as.numeric(as.character(sorted_predictions$pred))
      
      df_train_results$train_acc[i] <- round(train_acc$overall[1],4)
      
      # fit tuned model on full dataset
      
      fitted_final <- caret::train(y = factor(labeled_dtm_nb[,1]), x = labeled_dtm_nb[,-1], method = method_train, metric = "Accuracy",
                                   tuneGrid = fitted$bestTune, trControl = ctrl_final)
      
      final_model_final <- fitted_final # recommended not to work w/ $finalModel
      repeated_acc <- caret::confusionMatrix(fitted_final$pred$pred, fitted_final$pred$obs)
      
      final_model_list_final[[i]] <- final_model_final
      models_final[[i]] <- fitted_final
      
      df_train_results$repeated_acc[i] <- round(repeated_acc$overall[1],4)
      
    }
    
  }
)

# Save models

df_train_results$final_model <- final_model_list
df_train_results$model <- models
df_train_results$tuned_parameters <- tuned_parameters
df_train_results$runtime <- round(unlist(runtime),2)

# Save tuned parameters

parameters <- data.frame(df_train_results$tuned_parameters[[1]]$layer1,
                         df_train_results$tuned_parameters[[2]]$k,
                         df_train_results$tuned_parameters[[3]]$fL,
                         df_train_results$tuned_parameters[[4]]$mtry,
                         df_train_results$tuned_parameters[[5]]$C)
colnames(parameters) <- c("ANN_layer1", "kNN_k", "NB_fL", "RF_mtry", "SVM_C")

# Compute standard errors

std <- function(x) sd(x)/sqrt(length(x))
std_dev <- vector(mode="numeric", length=0)
std_err <- vector(mode="numeric", length=0)

for(l in 1:length(set_of_models)) {
  
  std_dev[l] <- sd(final_model_list_final[[l]]$resample$Accuracy)
  std_err[l] <- std(final_model_list_final[[l]]$resample$Accuracy)
  
}

df_train_results$std_dev <- round(std_dev,4); df_train_results$std_err <- round(std_err,4)

# Create accuracy df

df_train_accuracies <- as.data.frame(matrix(nrow = ctrl_final$number * ctrl_final$repeats, ncol = length(set_of_models)))
colnames(df_train_accuracies) <- c("ANN", "kNN", "NB", "RF", "SVM")

for(m in 1:length(set_of_models)) {
  
  df_train_accuracies[,m] <- round(final_model_list_final[[m]]$resample$Accuracy,4)
  
}

# Make predictions

df_train_results$test_acc = NA
predictions <- as.data.frame(matrix(nrow = nrow(test_labeled), ncol = length(final_model_list)))
colnames(predictions) <- c("ANN", "kNN", "NB", "RF", "SVM")

for(j in 1:length(final_model_list)) {
  
  method_train <- model_parameter_grids$model[j]
  
  if(method_train != "nb") {
    
    start_time = Sys.time()
    pred_i <- predict(final_model_list[[j]], test_labeled[, -1], type = "raw")
    end_time <- Sys.time()
    
  }
  else {  
    
    start_time = Sys.time()
    pred_i <- predict(df_train_results$final_model[[j]], test_labeled_nb[, -1], type = "raw")
    end_time <- Sys.time()
    
  }
  
  time_fitted <- end_time - start_time
  df_train_results$prediction_time[j] <- round(time_fitted,2)
  
  test_acc <- caret::confusionMatrix(pred_i, test_labeled[,1])
  df_train_results$test_acc[j] <- round(test_acc$overall[1],4)
  predictions[, j] = pred_i
  
}

df_train_results$rank <- rank(1/df_train_results$test_acc)

# Conduct t-test against winning method

t_test <- vector(mode="numeric", length=0)

for(n in 1:length(set_of_models)) {
  
  t_test[n] <- tsum.test(df_train_results$test_acc[which.min(df_train_results$rank)], 
                         df_train_results$std_dev[which.min(df_train_results$rank)], 50,
                         df_train_results$test_acc[n], df_train_results$std_dev[n], 50)$p.value
  
}

df_train_results$t_test <- round(t_test,4)
df_train_results$not_sign <- ifelse(df_train_results$t_test < 0.05, 0, 1)

# Run majority vote

df_train_ensemble <- cbind(test_labeled[,1], predictions)
colnames(df_train_ensemble)[1] <- "class"

m_vote <- apply(df_train_ensemble[,-1], 1, function(x) names(which.max(table(x))))
vote_acc <- round(caret::confusionMatrix(m_vote, df_train_ensemble[,1])$overall[1],4)

# Train stacking ensemble

model_2 <- caret::train(as.factor(class)~., data = train_predictions, method="mlpWeightDecayML", trControl = ctrl_tune)
predictions_numeric <- as.data.frame(unlist(apply(predictions, 2, function(x) as.numeric(as.character(x)))))

stacking_predictions <- predict(model_2, newdata = predictions_numeric)
stacking_acc <- round(caret::confusionMatrix(stacking_predictions, test_labeled[,1])$overall[1],4)

# Save ensemble results

ensemble <- data.frame(vote_acc, stacking_acc); ensemble

# Export results to Excel file

results_cols <- c("train_acc", "test_acc", "rank", "t_test", "not_sign", "std_dev", "std_err", "runtime", "prediction_time")
results <- df_train_results[,results_cols]
rownames(results) <- c("ANN", "kNN", "NB", "RF", "SVM")
results

l <- list("results" = results, "accuracies" = df_train_accuracies, "parameters" = parameters, "descriptives" = descriptives, "backup" = ensemble)
write.xlsx(l,paste0(Sys.Date(), "_results", ".xlsx"), col.names = TRUE, row.names = TRUE)

# Plot results as bar chart

ggplot(results, aes(x=rownames(results), y=test_acc)) + 
  geom_bar(position=position_dodge(), stat="identity", fill = "#4285f4", size=.3) +
  geom_errorbar(aes(ymin=test_acc-2*std_err, ymax=test_acc+2*std_err), size=.3,
                width=.2, position=position_dodge(.9)) +
  geom_text(aes(label = sprintf("%.2f", test_acc), y= test_acc),  vjust = -2)+
  xlab("Method") +
  ylab("Accuracy (%)") +
  scale_y_continuous(limits = c(0,1)) +
  ggtitle(paste0("Accuracy per Method (", id, "), ", "N=", nrow(data))) +
  theme_classic()

ggsave(paste0(Sys.Date(), "_results", ".jpg"), 
       plot = last_plot(), width = 10, height = 10)

# Stop parallel processing ------------------------------------------------

stopCluster(cluster); registerDoSEQ()

# END 


## GENERAL COMMENT: Wrap parameters in variables so the code is mode readable/understandable
## Evaluation: Keep in mind the performance of specific classes: whether false positives / false negatives are particularly high 
# in cases where overall accuracy is improved beyond 80%

## Code is okay for a prototype
## When moving it into production, ideally we should break the code down into more functions with clearly defined inputs and outputs
## Max hasn't worked with hierachical models - investigate more myself
## ndcg: normalized discounted cumulative gain - use it when using a ordered list of reccommended categories



