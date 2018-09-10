rm(list = ls())


#load packages
library(caret)
library(cld2)



#functions
createDataFrame = function(df, p) {
  df = readRDS(file = "train.rds")
  sample_size = p
  df$main_category = factor(df$main_category)
  #detect language
  df$language = detect_language(df$body)
  #table(df$language=="en")
  df = subset(df, language=="en")
  
  ## Cleaning
  df$text = gsub("\\\\n", "", df$body)
  
  df = df[,c(4,2)]
  colnames(df)[2] = 'class'
  
  set.seed(111)
  
  subsample = createDataPartition(y = df$class, p = sample_size, list = FALSE)
  df = df[subsample,]
  return(df)
}


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


create_labelled_dtm = function(df1, corpus) {
  
  df = as.data.frame(cbind(df1$class, corpus_clean))
  df[,-1] = apply(df[,-1], 2, function(x) as.numeric(as.character(x)))
  colnames(df)[1] = "class_dv"
  return(df)
}


run_model = function(train, test) {
  
  ctrl_tune = trainControl(method = "repeatedcv", number = 3, repeats = 1, selectionFunction = "best", 
                           verboseIter = TRUE, classProbs = FALSE)
  grid_rf = expand.grid(mtry = 38)
  
  set_of_models = c("ranger")
  
  model_parameter_grids = as.data.frame(matrix(nrow = length(set_of_models), ncol = 2))
  colnames(model_parameter_grids) <- c("model", "parameter_grid")
  model_parameter_grids$model = set_of_models
 
  model_parameter_grids$parameter_grid = grid_rf
  test_predictions = as.data.frame(matrix(nrow = nrow(test), ncol = length(set_of_models) + 1))
  colnames(test_predictions) = c("class", "RF")
  test_predictions$class = test$class_dv
  

  
  fitted = caret::train(y = factor(train[,1]), x = train[,-1], method = "rf", metric = "Accuracy",
                        tuneGrid = grid_rf, trControl = ctrl_tune)
  
  test_predictions$RF = predict(fitted, test)
  
  return(caret::confusionMatrix(test_predictions$RF, test_predictions$class))
}

