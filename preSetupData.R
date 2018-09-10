
library(readr)


train = read_csv("~/NLP-CX/train.csv")


train$case_id = NULL

train = train[!is.na(train$body),]


categories = c("3. [POST booking] Existing Booking", "1. [PRE booking] Activity Information", "12. Supplier", "5. Cancellation",
"4. [POST booking] Booking Modification", "6. Complaint", "2. [PRE booking] Booking Process", "7. Finance Support")

train = train[train$main_category %in% categories,]

View(train)

train$main_category = factor(train$main_category)



saveRDS(object = train, file = "train.rds")



