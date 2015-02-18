# Enabling Multi Core for modeling processing
library(doMC)
registerDoMC(cores = 2)

#Loading used libraries
library(caret); library(rattle)
library(rpart); library(klaR)
library(randomForest); library(gbm)

#setting the seed for reproducible computation
set.seed(12345)

#setting the working directory folder
setwd("~/Developer/Data Science Specialization/Practical Machine Learning/Project")

# loading both testing and training dataset (considering both files were already downloaded)
trainFile <- "./pml-training.csv"
training <- read.csv(file=trainFile, header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

testFile <- "./pml-testing.csv"
testing <- read.csv(file=testFile, header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

# Starting the Cleaning Process
nzvCol <- nearZeroVar(training)
training <- training[,-nzvCol]

# Since we have lots of variables, remove any with NA's or have empty strings, and the one's that are not predictors variables
filterData <- function(idf) {
    idx.keep <- !sapply(idf, function(x) any(is.na(x)))
    idf <- idf[, idx.keep]
    idx.keep <- !sapply(idf, function(x) any(x==""))
    idf <- idf[, idx.keep]
    
    # Remove the columns that aren't the predictor variables
    col.rm <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                "cvtd_timestamp", "new_window", "num_window")
    idx.rm <- which(colnames(idf) %in% col.rm)
    idf <- idf[, -idx.rm]
    return(idf)
}

training <- filterData(training)
finalTrainingDS <- training
dim(finalTrainingDS)

# Now let's perform the same cleaning process to the testing dataset as well
nzvCol <- nearZeroVar(testing)
testing <- testing[,-nzvCol]
testing <- filterData(testing)
finalTestingDS <- testing
dim(finalTestingDS)

# Partitioning the dataset 
inTrain <- createDataPartition(y=finalTrainingDS$classe, p=0.85, list=FALSE)
newTraining <- finalTrainingDS[inTrain, ]
newTesting <- finalTrainingDS[-inTrain, ]
dim(newTraining); dim(newTesting)

#Some fitting params for Cross Validations
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=FALSE)

# Performing the Analysis Without the PCA
# Decision Trees
modelFitTree <- train(classe ~ . , method="rpart", data=newTraining, trControl=fitControl)
cm_tree <- confusionMatrix(newTesting$classe, predict(modelFitTree, newdata=newTesting))
cm_tree$overall

# Linear Discriminant Analysis
modelFitLDA <- train(classe ~ ., method="lda", data=newTraining, trControl=fitControl)
cm_lda <- confusionMatrix(newTesting$classe, predict(modelFitLDA, newdata=newTesting))
cm_lda$overall

#Naive Baeyes
modelFitNB <- train(classe ~ ., method="nb", data=newTraining, trControl=fitControl)
cm_nb <- confusionMatrix(newTesting$classe, predict(modelFitNB, newdata=newTesting))
cm_nb$overall

#Random Forest
# as this modeling takes some time, I'll be saving it for later use.
modelFitRF <- train(classe ~ . , method="rf", data=newTraining)
saveRDS(modelFitRF, "rfmodel.RDS")
cm_rf <- confusionMatrix(newTesting$classe, predict(modelFitRF, newdata=newTesting))
cm_rf$overall

#Generalized Boosted Regression Modeling
modelFitGBM <- train(classe ~ ., method="gbm", data=newTraining, trControl=fitControl)
cm_gbm <- confusionMatrix(newTesting$classe, predict(modelFitGBM, newdata=newTesting))
cm_gbm$overall

## Performing the same analysis, but pre-processing the data with PCA
prePro <- preProcess(newTraining[,-53], method="pca", tresh=0.99)
newTrainingPCA <- predict(prePro,newTraining[,-53])
newTestingPCA <- predict(prePro,newTesting[,-53])

# Decision Trees
modelFitTreePCA <- train(newTraining$classe ~ . , method="rpart", trControl=fitControl, data=newTrainingPCA)
cm_tree_pca <- confusionMatrix(newTesting$classe, predict(modelFitTreePCA, newdata=newTestingPCA))
cm_tree_pca$overall

# Linear Discriminant Analysis
modelFitLDAPCA <- train(newTraining$classe ~ . , method="lda", trControl=fitControl, data=newTrainingPCA)
cm_lda_pca <- confusionMatrix(newTesting$classe, predict(modelFitLDAPCA, newdata=newTestingPCA))
cm_lda_pca$overall

# Naive Baeyes
modelFitNBPCA <- train(newTraining$classe ~ . , method="nb", trControl=fitControl, data=newTrainingPCA)
cm_nb_pca <- confusionMatrix(newTesting$classe, predict(modelFitNBPCA, newdata=newTestingPCA))
cm_nb_pca$overall

# Random Forest
modelFitRFPCA <- train(newTraining$classe ~ . , method="rf", data=newTrainingPCA)
cm_rf_pca <- confusionMatrix(newTesting$classe, predict(modelFitRFPCA, newdata=newTestingPCA))
cm_rf_pca$overall

# Generalized Boosted Regression Modeling
modelFitGBMPCA <- train(newTraining$classe ~ . , method="gbm", trControl=fitControl, data=newTrainingPCA)
cm_gbm_pca <- confusionMatrix(newTesting$classe, predict(modelFitGBMPCA, newdata=newTestingPCA))
cm_gbm_pca$overall

# Analyzing the results
resultingDataTable <- rbind(cm_tree$overall, cm_tree_pca$overall, 
                       cm_lda$overall, cm_lda_pca$overall, 
                       cm_nb$overall, cm_nb_pca$overall, 
                       cm_rf$overall, cm_rf_pca$overall, 
                       cm_gbm$overall, cm_gbm_pca$overall)
rownames(resultingDataTable) <- c("Tree", "Tree w/ PCA", 
                             "LDA", "LDA w/ PCA",
                             "Naive Baeyes", "Naive Baeyes w/ PCA",
                             "Random Forest", "Random Forest w/ PCA",
                             "GBM", "GBM w/ PCA")

resultingDataTable

From the resulting table we can see the Random Forest algorithm yields a better result, and that using Principal Component Analysis, 
with a threshold of 99% of the variance would descrease of accuracy in about (aprox) 2% points.

# Out of Sample Error 
Our out-of-sample error can be found using the 1 - Testing Accurary, as evaluated below.

oos_error <- 1 - cm_rf$overall[1]
print(paste("Out of Error Sample", oos_error * 100, "%")
)

For the sake of curiosity, lets check the importance of each variable at the final model.     
varImp(modelFitRF)



# Generating files for the 2nd part of the assignment
# This uses the code supplied by the class instructions

answers <- predict(modelFitRF, newdata=testing)
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
      filename = paste0("problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(answers)
