
source("quickRun2.R")

#Specific the sample and test split
sample_size = 0.06
train_test_split = 0.8

#Pre-process data
data = createDataFrame(df = data, p = sample_size)
data$usable_text = sapply(data$text, function(x) iconv(enc2utf8(x), sub = "byte"))
corpus_clean = pre_process_data(data)
labeled_dtm = create_labelled_dtm(df1 = data, corpus = corpus_clean)

#Create test and training sets
set.seed(128) 
trainIndex = createDataPartition(data$class, p = train_test_split, list = FALSE)
train_labeled = labeled_dtm[trainIndex,]
test_labeled = labeled_dtm[-trainIndex,]

#Run the model
run_model(train = train_labeled, test = test_labeled)

