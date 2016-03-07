import sys
sys.path.append("..")
import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
import collections
import theautil

# setup logging
logging.basicConfig(stream = sys.stderr, level=logging.INFO)

# general setting
mupdates = 1000

# load training data and bigger test data
data = np.loadtxt("twosensors.csv", delimiter=",")
testData = np.loadtxt("twosensors2.csv",delimiter=",")
inputs  = data[0:,0:11].astype(np.float32)
outputs = data[0:,11:12].astype(np.int32)
testInputs  = testData[0:,0:11].astype(np.float32)
testOutputs = testData[0:,11:12].astype(np.int32)

# shuffle the dataset for further split
theautil.joint_shuffle(inputs,outputs)

# split the training dataset and using 90% for training and 10% as a small test dataset 
train_and_valid, test = theautil.split_validation(90, inputs, outputs)

# the final training data is 90%*90% of the original dataset 
train, valid = theautil.split_validation(90,train_and_valid[0],train_and_valid[1])

def linit(x):
    return x.reshape((len(x),))

# reshape
train = (train[0],linit(train[1]))
valid = (valid[0],linit(valid[1]))
test = (test[0],linit(test[1]))
testNew  = (testInputs ,linit(testOutputs))

# build the model
#net = theanets.Classifier([11,(100,'softplus'),(500,'softplus'),(1000,'softplus'),(500,'softplus'),(250,'softplus'),(100,'softplus'),(2,'softplus')])
net = theanets.Classifier([11,(100,'softplus'),(500,'softplus'),(1000,'softplus'),(500,'softplus'),2])
#net = theanets.Classifier([11,50,500,500,200,50,2])
net.train(train, valid, algo='layerwise',max_updates=mupdates)

# save the model
print "Finsh and Save model"
net.save("models/model([11,(100,'softplus'),(500,'softplus'),(1000,'softplus'),(500,'softplus'),2]]|algo=layerwise,max_updates=1000")

# print out the results
print "Learner on the test set"
classify = net.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])
print "Learner on the Larger test set"
classify = net.classify(testNew[0])
print "%s / %s " % (sum(classify == testNew[1]),len(testNew[1]))
print collections.Counter(classify)
print theautil.classifications(classify,testNew[1])



