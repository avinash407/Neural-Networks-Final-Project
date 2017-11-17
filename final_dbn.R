library(mxnet)
library(mlbench)
library(darch)

#setwd('C:/Users/pruth/Desktop')

review <- read.csv('SparseRepresentation.csv')

# review <- review[,3:7]
# review
review[,580] = as.numeric(review[,580])-1
review

trainData <- review[1:2000,]
testData <- review[2001:5000,]

train.x <- data.matrix(trainData[,1:579])
train.x
train.y <- trainData[,580]
train.y

test.x <- data.frame(testData[,1:579])
test.x
test.y <- testData[,580]
test.y


# darch <- darch(result ~ .,trainData)
# print(darch)
# predictions <- predict(darch, newdata = test.x, type ='Class')

#preds =predict(model,test.x)
#preds
#install.packages("deepnet")
library(deepnet)

dnn <- dbn.dnn.train(train.x, train.y, hidden = c(3,3),activationfun = "sigm", output = "sigm", numepochs = 100, cd=2)
summary(dnn)
err.dnn <- nn.test(dnn, test.x, test.y)
err.dnn
yy.dnn <- nn.predict(dnn, test.x)
summary(yy.dnn)
head(yy.dnn)



#install.packages("Lahman")
#install.packages("SDMTools")
library(Lahman)
library(SDMTools)
accuracy(test.y,yy.dnn,0.5)


confusion.matrix(test.y,yy.dnn,threshold=0.5)


confusion.matrix(test.y, yy.dnn)



write.csv(yy.dnn,'DBN_Out.csv')

m <- round(yy.dnn)
table( yy.dnn, test.y)
#confusionMatrix(table( yy.dnn, test.y))

plot(dnn)
getwd()


