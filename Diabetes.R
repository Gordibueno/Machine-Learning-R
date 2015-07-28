# In this project, use the diabetes data in Efron et al. (2003) to examine the effects of 
# ten baseline predictor variables [age, sex, body mass index (bmi), average blood pressure (map), 
# and six blood serum measurements (tc, ldl, hdl, tch, ltg, glu)] on a quantitative measure of disease 
# progression one year after baseline. There are 442 diabetes patients in this data set. The data are 
# available in the R package “lars.” Employ several machine learning techniques using the 
# diabetes data to fit linear regression, ridge regression and lasso models. You must also incorporate 
# best subset selection and cross-validation techniques.

# Load diabetes data
install.packages("lars")
library(lars)
data(diabetes)
?diabetes
data.all <- data.frame(cbind(diabetes$x, y = diabetes$y))

# EDA
install.packages("psych")
library(psych)
attach(data.all)
str(data.all)
sum(is.na(data.all)) # no missing values
summary(data.all) # predictor variables have been standardized, 0 mean
pairs(data.all)
round(cor(data.all),2) 
pairs.panels(data.all)

# Partition data into two group (75/25)
n = dim(data.all)[1] # sample size 442
set.seed(1306)

test = sample(n, round(n/4)) # randomly sample 25% test
data.train = data.all[-test,]
data.test <- data.all[test,]
x = model.matrix(y ~ ., data = data.all)[,-1] # define predictor matrix

x.train <- x[-test,] # define training predictor matrix
x.test <- x[test,] # define test predictor matrix
y <- data.all$y # define response variable
y.train <- y[-test] # define training response variable
y.test <- y[test] # define test response variable
n.train <- dim(data.train)[1] # training sample size = 332
n.test <- dim(data.test)[1] # test sample size = 110

##### 1. Least Squares Regression w/All Predictors  #####
x.train.df = data.frame(x.train)
lm.fit=lm(y.train~., data=x.train.df)
summary(lm.fit)

x.test.df = data.frame(x.test)
lm.model=predict(lm.fit, newdata=x.test.df)
summary(lm.model)
mean((y.test - lm.model)^2) # mean squared error
sd((y.test - lm.model)^2) # standard error

##### 2. Best Subset Selection Using BIC #####
library(leaps)
regfit.full=regsubsets(y.train~., x.train.df, nvmax=10)
reg.summary = summary(regfit.full)
reg.summary
reg.summary$bic
reg.summary$rsq
reg.summary$adjr2
reg.summary$cp
par(mfrow=c(1,3))
plot(reg.summary$bic, xlab="Number of Variables", ylab="BIC", type="l")
plot(reg.summary$adjr2, xlab="Number of Variables", ylab="Adj R2", type="l")
plot(reg.summary$cp, xlab="Number of Variables", ylab="CP", type="l")
which.min(reg.summary$bic)
coef(regfit.full,6)

regfit.sub = lm(y.train ~ sex + bmi + map + tc + tch + ltg, data=x.train.df)
names(regfit.sub)
summary(regfit.sub)

regfit.sub.pred = predict(regfit.sub, newdata = x.test.df)
mean((y.test - regfit.sub.pred)^2)
sd((y.test - regfit.sub.pred)^2)

# predict method for best subset selection
predict.regsubsets = function(object, newdata, id, ...) {
    form = as.formula(object$call[[2]])
    mat = model.matrix(form, newdata)
    coefi = coef(object, id = id)
    mat[, names(coefi)] %*% coefi
}

#####  3. Best Subset Selection Using 10-Fold CV  #####
k=10
set.seed(1306)
folds = sample(1:k, nrow(data.train), replace = TRUE)
cv.errors = matrix(NA,k,10,dimnames = list(NULL, paste(1:10)))

for (j in 1:k){
    best.fit = regsubsets(y~., data=data.train[folds!=j,], nvmax =10)
    for (i in 1:10){
        pred = predict(best.fit, data.train[folds==j,], id=i)
        cv.errors[j,i] = mean((data.train$y[folds==j]-pred)^2)
    }
}

mean.cv.errors = apply(cv.errors,2,mean)
sd.cv.errors = apply(cv.errors,2,sd)
mean.cv.errors
sd.cv.errors
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')
# six-model variable also shown to be best here

##### Ridge Regression Using 10-Fold CV #####
library(glmnet)
grid = 10^seq(10,-2,length=100)
ridge.mod <- glmnet(x.train,y.train,alpha=0,lambda=grid)

set.seed(1306)
ridge.mod.out <- cv.glmnet(x.train,y.train,alpha=0)
plot(ridge.mod.out)
bestlam.ridge = ridge.mod.out$lambda.1se
bestlam.ridge

ridge.pred = predict(ridge.mod, s = bestlam.ridge, newx=x.test)
mean((ridge.pred - y.test)^2)
sd((y.test - ridge.pred)^2)

ridge.out = glmnet(x, y, alpha=0)
predict(ridge.out, type="coefficients", s=bestlam.ridge)[1:11,]

##### Lasso Using 10-Fold #####
lasso.fit <- glmnet(x.train,y.train,alpha=1,lambda=grid)
plot(lasso.fit)

set.seed(1306)
cv.lasso.out <- cv.glmnet(x.train,y.train,alpha=1)
bestlam.cv = cv.lasso.out$lambda.1se
bestlam.cv

lasso.pred = predict(lasso.fit, s = bestlam.cv, newx=x.test)
mean((lasso.pred - y.test)^2) 
sd((lasso.pred - y.test)^2) 

lasso.out = glmnet(x,y,alpha=1, lambda=grid)
lasso.coef = predict(lasso.out, type="coefficients", s=bestlam.cv)[1:11,]
lasso.coef
