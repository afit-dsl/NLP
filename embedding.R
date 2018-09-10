rm(list = ls())

### LOAD AND PRE-PROCESS
require(text2vec)

gyg = import("training_set.csv")
gyg[,1] = NULL
gyg$class = substr(gyg[,2],1,1)
gyg$language = detect_language(gyg$body)
gyg_engl = subset(gyg, language=="en")
gyg_engl$text <- gsub("\\\\n", "", gyg_engl$body)
#install.packages("text2vec")

### TAKE A STRATIFIED SAMPLE
set.seed(111)
subsample = createDataPartition(y = gyg_engl$class, p = 0.1, list = FALSE)
wiki= gyg_engl[subsample,]
wiki = wiki[,5]

rm(gyg)
rm(gyg_engl)

### WORD EMBEDDING
# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 5L)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)



glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
glove$fit_transform(tcm, n_iter = 20)

word_vectors <- glove$get_word_vectors()


berlin <- word_vectors["paris", , drop = FALSE] - 
  word_vectors["france", , drop = FALSE] + 
  word_vectors["germany", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = berlin, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 5)

