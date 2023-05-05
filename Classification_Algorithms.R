library(MASS)
library(smallstuff)
library(boot)
library(class)
library(leaps)
library(broom)
library(glmnet)
library(data.table)
library(gam)
library(ggplot2)
#library(tidyr)

# The following code in based on the 9 steps mentioned in the final project 
# instructions file

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# 1)

data = fread('../data/HTRU_2.csv')
setDT(data)

# Our target variable represents the classification of whether a star is a pulsar
# (1) or not a pulsar (0), we will be using all columns in our analysis

summary(data)
# No missing values were found

# We only have one categorical variable, which is the target class
class(data$target_class)

# We can convert it to a factor as follows:
data[, target_class:=factor(target_class)]
class(data$target_class)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# 2)

# Splitting the data between training and testing,
# with 80% of the data used for training and 20% for testing

set.seed(42069)
te=sort(sample(nrow(data),round2(nrow(data)/5)))
PulsarTrain=data[-te,]
PulsarTest=data[te,]
dim(PulsarTest) # 3580 x 9
dim(PulsarTrain) # 14318 x 9

summary(PulsarTest$target_class)   # 334/3246
summary(PulsarTrain$target_class)  # 1305/13013

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# 3)

# Data Source:
# https://archive.ics.uci.edu/ml/datasets/HTRU2#


# The data set shared here contains 16,259 spurious examples caused by RFI/noise,
# and 1,639 real pulsar examples. These examples have all been checked by human annotators.
# We chose it because we are passionate about astronomy and physics.

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# 4) Creating all models and evaluating their performance

error_rates = c(rep(0, 7))

# LDA model - All predictors
ldamod=lda(target_class~., PulsarTrain)
predl=predict(ldamod)

mean(predl$class!=PulsarTrain$target_class)   # Training set: 2.49% error rate
set.seed(42069);CVerror(ldamod,5)                 #5-fold CV error rate=2.535
set.seed(42069);CVerror(ldamod,10)                #10-fold CV error rate=2.493

predl_test=predict(ldamod, PulsarTest)
mean(predl_test$class!=PulsarTest$target_class)
# Test set: 2.63% error rate

error_rates[1] = mean(predl_test$class!=PulsarTest$target_class)

#-------------------------------------------------------------------------------

# QDA model - All predictors
qdamod=qda(target_class~., PulsarTrain)
predq=predict(qdamod)

mean(predq$class!=PulsarTrain$target_class) # Training set: 3.23% error rate
set.seed(42069);CVerror(qdamod,5)                 #5-fold CV error rate=3.255
set.seed(42069);CVerror(qdamod,10)                #10-fold CV error rate=3.248

predq_test=predict(qdamod, PulsarTest)
mean(predq_test$class!=PulsarTest$target_class)
# Test set: 3.29% error rate

error_rates[2] = mean(predq_test$class!=PulsarTest$target_class)

#-------------------------------------------------------------------------------

# Logistic Regression - All predictors

gmod=glm(target_class~., binomial, PulsarTrain)

logistErrorRate(gmod, PulsarTrain)  # 1.98% Training error rate
set.seed(42069);CVerror(gmod,5)                 #5-fold CV error rate=2.018
set.seed(42069);CVerror(gmod,10)                #10-fold CV error rate=2.011

logistErrorRate(gmod, PulsarTest)
# 2.291% Testing error rate

summary(gmod)
# This shows that 2 predictors are not significant:
# Excess kurtosis of the DM-SNR curve, and 
# Skewness of the DM-SNR curve

# Let's see how it performs without these predictors:

#-------------------------------------------------------------------------------

# Logistic Regression - Reduced model 1

gmod2=glm(target_class~., binomial, PulsarTrain[,-"Excess kurtosis of the DM-SNR curve"])

logistErrorRate(gmod2, PulsarTrain)                # 1.99% Training error rate
set.seed(42069);CVerror(gmod2,5)                 #5-fold CV error rate=2.025
set.seed(42069);CVerror(gmod2,10)                #10-fold CV error rate=2.032

logistErrorRate(gmod2, PulsarTest)
# 2.291% Testing error rate
# Removing this predictor did not impact our testing error rate

error_rates[3] = logistErrorRate(gmod2, PulsarTest)$errorRate/100

#-------------------------------------------------------------------------------

# Logistic Regression - Reduced model 2

gmod3=glm(target_class~., binomial, PulsarTrain[,-c("Excess kurtosis of the DM-SNR curve",
                                                   "Skewness of the DM-SNR curve")])

logistErrorRate(gmod3, PulsarTrain)                # 1.99% Training error rate
set.seed(42069);CVerror(gmod3,5)                 #5-fold CV error rate=1.997%
set.seed(42069);CVerror(gmod3,10)                #10-fold CV error rate=1.997%


logistErrorRate(gmod3, PulsarTest)
# 2.37% Testing error rate

summary(gmod)
# All predictors are significant, testing accuracy is only slightly worse

#-------------------------------------------------------------------------------

# Ridge Regression

X1=model.matrix(target_class~.,PulsarTrain)[,-1]
X1t=model.matrix(target_class~.,PulsarTest)[,-1]

rmod=glmnet(X1,PulsarTrain$target_class, "binomial", alpha=0)
rmod$lambda
lambdas=c(300:2,300:0/1000)
rmod=glmnet(X1,PulsarTrain$target_class, family = "binomial", alpha=0,lambda=lambdas)

set.seed(42069)
(lamr=cv.glmnet(X1,PulsarTrain$target_class, family = "binomial", alpha=0,lambda=lambdas)$lambda.min)
# Lambda is 0

#Test error rate for ridge regression
pred=predict(rmod,X1t,s=lamr,type="r")
yhat=rep(levels(PulsarTest$target_class)[1],nrow(PulsarTest))
yhat[pred[,1]>.5]=levels(PulsarTest$target_class)[2]
mean(PulsarTest$target_class!=yhat) 

# Error Rate: 2.291%

error_rates[4] = mean(PulsarTest$target_class!=yhat) 

#-------------------------------------------------------------------------------

# Lasso Regression
lassomod=glmnet(X1,PulsarTrain$target_class, family="binomial", lambda=lambdas)
set.seed(42069)
(laml=cv.glmnet(X1,PulsarTrain$target_class, family="binomial", lambda=lambdas)$lambda.min)

#Test error rate for lasso regression
pred=predict(lassomod,X1t,s=laml,type="r")
yhat=rep(levels(PulsarTest$target_class)[1],nrow(PulsarTest))
yhat[pred[,1]>.5]=levels(PulsarTest$target_class)[2]
mean(PulsarTest$target_class!=yhat) 
# Error Rate: 2.291%

error_rates[5] = mean(PulsarTest$target_class!=yhat) 

#-------------------------------------------------------------------------------

# K nearest neighbors Models

(Ks=c(1,seq(1,20,by=5),seq(25,round2(nrow(PulsarTrain)/29),by=100)))

knntrain=NULL;knn5err=NULL;knn10err=NULL
for (i in (1:length(Ks))) {
  set.seed(42069)
  knntrain[i]=mean(knn(PulsarTrain[,1:8],PulsarTrain[,1:8],PulsarTrain$target_class,k=Ks[i])!=PulsarTrain$target_class)*100
  set.seed(42069);knn5err[i]=CVerrorknn(PulsarTrain[,1:8],PulsarTrain$target_class,K=Ks[i],k=5)
  set.seed(42069);knn10err[i]=CVerrorknn(PulsarTrain[,1:8],PulsarTrain$target_class,K=Ks[i],k=10)
}
# Note: grab a coffee while this runs (~5min)

par(mfrow=c(2,2))
plot(knntrain~Ks,type='b',main="Training",xlab="K",ylab="% Error Rate")
plot(knn5err~Ks,type='b',main="5-CV",xlab="K",ylab="% Error Rate")
plot(knn10err~Ks,type='b',main="10-CV",xlab="K",ylab="% Error Rate")
par(mfrow=c(1,1))

Ks[c(which.min(knntrain),which.min(knn5err),which.min(knn10err))]
round2(c(min(knntrain),min(knn5err),min(knn10err)),2)

# Our best k is 11, which has a training error rate of 2.67%

set.seed(42069)
yhat1=knn(PulsarTrain[,1:8],PulsarTest[,1:8],PulsarTrain$target_class,11,prob=T)
mean(yhat1!=PulsarTest$target_class) 

# Testing error rate: 2.598%

error_rates[6] = mean(yhat1!=PulsarTest$target_class) 

#-------------------------------------------------------------------------------

# GAMs
gamod=gam(target_class~., data=PulsarTrain, family = "binomial")
summary(gamod)

pred=predict(gamod,PulsarTest[,-'target_class'])
yhat=rep(levels(PulsarTest$target_class)[1],nrow(PulsarTest))
yhat[pred>.5]=levels(PulsarTest$target_class)[2]
mean(PulsarTest$target_class!=yhat) 
# Test error 2.430%

error_rates[7] = mean(PulsarTest$target_class!=yhat)


# GAMs - Removing all of the insignificant predictors
# We removed each predictor 1 by 1 according to their significance, the resulting
# model has all predictors significant

gamod=gam(target_class~., data=PulsarTrain[,-c('Skewness of the DM-SNR curve',
                                               'Mean of the DM-SNR curve',
                                               'Standard deviation of the integrated profile',
                                               'Excess kurtosis of the DM-SNR curve',
                                               'Standard deviation of the DM-SNR curve')], family = "binomial")
summary(gamod)

pred=predict(gamod,PulsarTest[,-'target_class'])
yhat=rep(levels(PulsarTest$target_class)[1],nrow(PulsarTest))
yhat[pred>.5]=levels(PulsarTest$target_class)[2]
mean(PulsarTest$target_class!=yhat) 
# Test error 2.514%


#-------------------------------------------------------------------------------

# Dummy Classifier

table(PulsarTrain$target_class)
# We will utilize "0" as our dummy classifier

table(PulsarTest$target_class)
# 3246 negative examples and 334 positive examples

# Testing Error Rate
(1-(3246/(3246+334)))
# 9.33%


#-------------------------------------------------------------------------------

# Final Remarks

# -All our models outperform the dummy classifier
# -Logistic Regression (Full and Reduced 1 Models) had the best performance (lowest test ER)
# -Our Lasso and Ridge Regression models had a best lambda of 0, therefore defaulting to logistic regression
# -The GAM model was our second best model, followed by LDA
# -It is important to note that our reduced GAM model, which used only 3 predictors 
# (mean of the integrated profile, excess kurtosis of IP, skeweness of IP), was still able to
# obtain significant high performance

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# 5) Picking our best model and running its diagnosis

# We chose the logistic regression to be our model due to its lower testing error
# rate (2.291%). We will now perform a deeper analysis on its results

logistErrorRate(gmod2, PulsarTest)
ler=logistErrorRate(gmod2, PulsarTest)
lr=ler$result

lr[2,2]/lr[3,2]   #Sensitivity=.820
# This means that we correctly identified 82.0% of pulsar on the testing set

lr[1,1]/lr[3,1]   #Specificity=.993
# This means that we correctly identified 99.3% of all non Pulsar objects on the testing

lr[2,1]/lr[3,1]   #False positive rate=.007
lr[1,2]/lr[3,2]   #False negative rate=.179
lr[2,2]/lr[2,3]   #Precision=.926

# Intrepretation of the coefficients

summary(gmod2)$coefficients

# The three coefficients with highest magnitude are:
# Excess kurtosis of the integrated profile (6.76), 
# Skewness of the integrated profile (-0.64)
# Mean of the DM-SNR curve (-0.0325)


#-------------------------------------------------------------------------------

# Make plot to visualize performance of all models

models = c("LDA", "QDA", "Logistic", "Ridge", "Lasso", "KNN - k=11", "GAM")
error_rates = 100*error_rates

df = data.frame(models, error_rates)
df$models = reorder(df$models, df$error_rates)

ggplot(df, aes(x = models, y = error_rates)) +
  geom_bar(stat = "identity", fill = "blue") +
  ggtitle("Error Rates by Model") +
  xlab("Model") +
  ylab("Error Rate (in %)") + 
  theme(panel.background = element_rect(fill = "white"))

