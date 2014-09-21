Practical Machine Learning Course Project 
========================================================
By Megan Williams 

**Background:**

    Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:
    
        Class A: exactly according to the specification (proper execution)
        
        Class B: throwing the elbows to the front (common mistake)
        
        Class C: lifting the dumbbell only halfway (common mistake)
        
        Class D: lowering the dumbbell only halfway (common mistake)
        
        Class F: Throwing the hips to the front (common mistake)

**Goal of this Project:**
The goal of this project is to predict the manner in which the exercise was performed (i.e., Class A, B, C, D, or F). This report will describe the following: 

    -How I built my model
    
    -How I used cross validation
    
    -What I think the expected out of sample error is
    
    -Explanation for my choices

**First, I will load the appropriate packages**

```r
library(AppliedPredictiveModeling)
library(caret)
```

```
## Loading required package: ggplot2
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```
**Next, I will load and examine the data**

```r
rm(list = ls(all = TRUE))

setwd('/Users/meganwilliams/Desktop/MachineLearning')

Training = read.csv(file="pml-training.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings = c('NA','','#DIV/0!'))
Testing = read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings = c('NA','','#DIV/0!'))

Training$classe = as.factor(Training$classe)  

dim(Training)
```

```
## [1] 19622   160
```

```r
dim(Testing)
```

```
## [1]  20 160
```

```r
summary(Training$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
**Next, I will get rid of any  missing values or unnecessary variables**

```r
NAs = apply(Training,2,function(x) {sum(is.na(x))}) 
Training = Training[,which(NAs == 0)]
NAs = apply(Testing,2,function(x) {sum(is.na(x))}) 
Testing = Testing[,which(NAs == 0)]
```
**Next, I will work on preprocessing the variables**

```r
pre_Proc = which(lapply(Training, class) %in% "numeric")

pre_Obj = preProcess(Training[,pre_Proc],method=c('knnImpute', 'center', 'scale'))
train = predict(pre_Obj, Training[,pre_Proc])
train$classe = Training$classe

test = predict(pre_Obj,Testing[,pre_Proc])
```
**Non-zero Variables**

*Now, let's remove the non-zero variables*

```r
nzv = nearZeroVar(train,saveMetrics=TRUE)
train = train[,nzv$nzv==FALSE]

nzv = nearZeroVar(test,saveMetrics=TRUE)
test = test[,nzv$nzv==FALSE]
```
**Cross validations**

*Now, we must split the data into one set for training and one set for cross validation. The cross validation set will be used as the train control method for our model*

```r
set.seed(12031987)

inTrain = createDataPartition(train$classe, p = 3/4, list=FALSE)
training = train[inTrain,]
crossValidation = train[-inTrain,]
```
**Model**

*Next, we will create the Train model using Random Forest.*


```r
fit = train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )
```

```
## 
## Attaching package: 'e1071'
## 
## The following object is masked from 'package:Hmisc':
## 
##     impute
```

```r
save(fit,file="/Users/meganwilliams/Desktop/MachineLearning/fit.R")
```
**Accuracy**

*Next, let's check out the accuracy of the training set and the cross-validation set*

```r
##Training Set
train_Pred <- predict(fit, training)
confusionMatrix(train_Pred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
##Cross Validation Set
cross_Pred <- predict(fit, crossValidation)
confusionMatrix(cross_Pred, crossValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    3    0    0    0
##          B    2  944    2    0    0
##          C    0    2  852    3    0
##          D    0    0    1  801    3
##          E    1    0    0    0  898
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.995    0.996    0.996    0.997
## Specificity             0.999    0.999    0.999    0.999    1.000
## Pos Pred Value          0.998    0.996    0.994    0.995    0.999
## Neg Pred Value          0.999    0.999    0.999    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.174    0.163    0.183
## Detection Prevalence    0.284    0.193    0.175    0.164    0.183
## Balanced Accuracy       0.998    0.997    0.998    0.998    0.998
```
**Out of Sample Error**

*Next, we should calculate the out of sample error. We do this by subtracting 1 from the accuracy for predictions made against the cross-validation set. The out of sample error is low, suggesting that it is unlikely that the test samples will be classified incorrectly.*

```r
Out_of_Sample_Error = 1-.9965
Out_of_Sample_Error
```

```
## [1] 0.0035
```
*Results*

*Look at the predictions on the real testing set*

```r
test_Pred = predict(fit, test)
test_Pred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
