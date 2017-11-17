library(mxnet)
library(mlbench)
library(deepnet)

setwd('C:\\Users\\avina\\Desktop\\prudential data\\Final Project')

review <- read.csv('SparseRepresentation.csv')
#review <- review[,1:579]
names(review)
review[,580] = as.numeric(review[,580])
review[c(581:625)] <- 0
trainData <- review[1:3000,]
testData <- review[3001:5000,]
trainData <- data.matrix(trainData)
testData <- data.matrix(testData)
View(trainData)
train.x <- trainData[,]
train.x
train.y <- trainData[,580]

train.x <- t(train.x)
test <- t(testData)


table(train.y)

data <- mx.symbol.Variable('data')

#Convolution step
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20, pad=c(2,2))
#Flattening step
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
#Pooling step
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

#Second convolution
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

#First full convolution
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")

#Second Full convolution
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
#Loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

#Reshaping
train.array <- train.x
dim(train.array) <- c(25, 25, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(25, 25, 1, ncol(test))


#CPU
n.gpu <- 1
device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})
train.x




mx.set.seed(100)
tic <- proc.time()


model1 <- mx.model.FeedForward.create(lenet, X=train.array, y=as.array(train.y),
                                      ctx=device.cpu, 
                                      num.round=3, 
                                      array.batch.size=50,
                                      learning.rate=0.05, momentum=0.9, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      batch.end.callback=mx.callback.log.train.metric(100))



#Predictions
preds <- predict(model1, test.array)
pred.label <- max.col(t(preds)) - 1
library(caret)
confusionMatrix(pred.label, testData[,580])




write.csv(preds,'CNN.csv')




