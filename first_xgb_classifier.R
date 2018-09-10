
### 0. SETUP

rm(list = ls())


#getwd()
#setwd("C:/...")

## 0.1 Load Packages
library("rio")
library("cld2")
library("caTools")
library("tm")
library("ngram")
library("RWeka")
library("caret")
library("e1071")
library("RSNNS")
library("klaR")
library("kernlab")
library("randomForest")
library("xgboost")
library("plyr")
library("class")

## 0.2 Load Data
gyg = import("training_set.csv")



#----------------------------------------------------------------------------------------#
### 1. Data Cleaning and Pre-processing

##
gyg[,1] = NULL
gyg$class = substr(gyg[,2],1,1)
#str(gyg)

## Language
gyg$language = detect_language(gyg$body)
#table(gyg$language=="en")
gyg_engl = subset(gyg, language=="en")

## Cleaning
gyg_engl$text <- gsub("\\\\n", "", gyg_engl$body)



#----------------------------------------------------------------------------------------#
### 2. Feature Engineering

## Create a training and test dataset

set.seed(111)

subsample = createDataPartition(y = gyg_engl$class, p = 0.4, list = FALSE)
gyg_engl = gyg_engl[subsample,]


rm(gyg)



corpus_clean <- VCorpus(VectorSource(gyg_engl$text)); inspect(corpus_clean[5:8]); lapply(corpus_clean[5:8], as.character)
corpus_clean <- tm_map(corpus_clean, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

#as.character(corpus_clean[[1]])


BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))

# Create document-term-matrix (DTM)

dtm <- DocumentTermMatrix(corpus_clean, control = list(tokenize = BigramTokenizer, wordLengths=c(3,Inf), bounds = list(global = c(3,Inf))))

dtm <- weightBin(dtm)
inspect(dtm)



# Create labeled training data

gyg_engl <- as.data.frame(cbind(gyg_engl$class, as.matrix(dtm)))

# Rename label to "class" and convert to factor

colnames(gyg_engl)[1] <- c("class")
gyg_engl$class <- as.factor(gyg_engl$class)

# Change all other features to numerics

gyg_engl[,-1] = apply(gyg_engl[,-1], 2, function(x) as.numeric(as.character(x)))
str(gyg_engl)

# Create labeled training data for NB with factor features

gyg_engl_nb <- gyg_engl
gyg_engl_nb[,-1] <- lapply(gyg_engl_nb[, -1], factor)
str(gyg_engl_nb)


set.seed(545)
index = createDataPartition(y = gyg_engl$class, p = 0.7, list = FALSE)
train_labeled = gyg_engl[index,]
test_labeled = gyg_engl[-index,]

saveRDS(train_labeled, "train_labeled.RDS")
saveRDS(test_labeled, "test_labeled.RDS")


## Feature Selection




#----------------------------------------------------------------------------------------#
### 3. Model Building

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, selectionFunction = "best", 
                     verboseIter = TRUE, savePredictions = "final")
grid_xgb <- expand.grid(nrounds = c(50,100), lambda = 0.1, alpha = 0.1, eta = 0.3); grid_xgb
start_time = Sys.time()
m_xgb <- caret::train(class ~ ., data = train_labeled, method ="xgbLinear", metric = "Accuracy", 
                      tuneGrid = grid_xgb, trControl = ctrl)
end_time = Sys.time()
saveRDS(m_xgb, file = 'm_xgb.rds')

running_time_m_xgb = end_time - start_time 
running_time_m_xgb

#remove nas
#train_labeled = train_labeled[apply(!is.na(train_labeled), 1, any), ]


#m_ols = caret::train(x = train_labeled[,2:14183], y = train_labeled$class, 
#                     method = "glm",trControl = ctrl)


#require(nnet)
#start_time = Sys.time()

#m_log_multi <- multinom(class ~ ., data = train_labeled)
#end_time = Sys.time()
#saveRDS(m_log_multi, file = 'm_log_multi.rds')

#running_time_log_multi = end_time - start_time 
#running_time_log_multi
#----------------------------------------------------------------------------------------#
### 4. Prediction and Performance Evaluation
#----------------------------------------------------------------------------------------#

#test_labeled$predictions = predict(object = m_log_multi, newdata = test_labeled)

test_labeled$predictions = predict(object = m_xgb, newdata = test_labeled)


#confusion matrix
caret::confusionMatrix(test_labeled$predictions, test_labeled$class)


