# Kaggle Titanic 

test = read.csv("~/PREDICT 422/Kaggle/test.csv")

titanic.train = read.csv("~/PREDICT 422/Kaggle/titanic_train.csv")
View(titanic.train) # produces tabular view, unlike simple open option
str(titanic.train)
class(titanic.train)
attach(titanic.train)
table(Survived)
prop.table(table(Survived))
titanic.train$Survived = rep(0, 418)

# prepare file for submission
submit = data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "theyallperish.csv", row.names = FALSE)

# Titanic: The Gender-Class Model
summary(Sex)
prop.table(table(Sex, Survived))
prop.table(table(Sex, Survived),1) # 1 for row proportions, 2 for column
test$Survived = 0
test$Survived[test$Sex == 'female'] = 1
submit.gender = data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit.gender, file = "allbutwomen.csv", row.names = FALSE)

summary(titanic.train$Age)
titanic.train$Child = 0
titanic.train$Child[titanic.train$Age < 18] = 1
aggregate(Survived ~ Child + Sex, data = titanic.train, FUN=sum)
aggregate(Survived ~ Child + Sex, data = titanic.train, FUN=length)
aggregate(Survived ~ Child + Sex, data=titanic.train, FUN=function(x) {sum(x)/length(x)})

titanic.train$Fare2 = '30+'
titanic.train$Fare2[titanic.train$Fare < 30 & titanic.train$Fare >= 20] = '20-30'
titanic.train$Fare2[titanic.train$Fare < 20 & titanic.train$Fare >= 10] = '10-20'
titanic.train$Fare2[titanic.train$Fare < 10] = '<10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=titanic.train, FUN=function(x) {sum(x)/length(x)})
test$Survived = 0
test$Survived[test$Sex == 'female'] = 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] = 0

submit = data.frame(PassengerId = test$PassengerId, Survived = test$Survived)

submit.class = data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit.class, file = "classy.csv", row.names = FALSE)

library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare 
             + Embarked, data=titanic.train, method="class")
plot(fit)
text(fit)

# let's make some prettier ones
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library('rattle')
library(rpart.plot)
library(RColorBrewer)

par(mfrow=c(1,1))
fancyRpartPlot(fit)

Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "dtree.csv", row.names = FALSE)
head(Prediction)

# let's max out the default limits just for kicks
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=titanic.train,
             method="class", control=rpart.control(minsplit=2, cp=0))
fancyRpartPlot(fit)

fit_interactive <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=titanic.train,
             method="class", control=rpart.control(minsplit=2))
new.fit <- prp(fit_interactive,snip=TRUE)$obj
fancyRpartPlot(new.fit)

# Feature Engineering
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# something in a name?
train$Name[1]
test$Survived = NA
combine = rbind(train, test)
combine$Name <- as.character(combine$Name)
combine$Name[1]
strsplit(combine$Name[1], split='[,.]')
strsplit(combine$Name[1], split='[,.]')[[1]][2]
combine$Title <- sapply(combine$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combine$Title <- sub(' ', '', combine$Title) # strip white space
table(combine$Title)
combine$Title[combine$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combine$Title[combine$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combine$Title[combine$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combine$Title = factor(combine$Title)
combine$FamilySize <- combine$SibSp + combine$Parch + 1
combine$Surname <- sapply(combine$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combine$FamilyID <- paste(as.character(combine$FamilySize), combine$Surname, sep="")
head(combine$FamilyID)
combine$FamilyID[combine$FamilySize <= 2] <- 'Small'
table(combine$FamilyID)
famIDs <- data.frame(table(combine$FamilyID))
View(famIDs)
famIDs <- famIDs[famIDs$Freq <= 2,]
combine$FamilyID[combine$FamilyID %in% famIDs$Var1] = 'Small'
combine$FamilyID = factor(combine$FamilyID)

train <- combine[1:891,]
test <- combine[892:1309,]
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 
                 Title + FamilySize + FamilyID, data=train, method="class")
fancyRpartPlot(fit)
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "dtree_v2.csv", row.names = FALSE)

# Random Forests
summary(combine$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combine[!is.na(combine$Age),], method="anova")
combine$Age[is.na(combine$Age)] <- predict(Agefit, combine[is.na(combine$Age),])
which(combine$Embarked == '')
combine$Embarked[c(62,830)] = "S"
combine$Embarked = factor(combine$Embarked)
summary(combine$Fare)
which(is.na(combine$Fare))
combine$Fare[1044] = median(combine$Fare, na.rm=TRUE)
combine$FamilyID2 = combine$FamilyID
combine$FamilyID2 = as.character(combine$FamilyID2)
combine$FamilyID2[combine$FamilySize <=3] = 'Small'
combine$FamilyID2 = factor(combine$FamilyID2)

train <- combine[1:891,]
test <- combine[892:1309,]

set.seed(1)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, data=train, importance=TRUE, ntree=2000)
varImpPlot(fit)

Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "rforest.csv", row.names = FALSE)

install.packages('party')
library(party)
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "cond_rforest.csv", row.names = FALSE)
