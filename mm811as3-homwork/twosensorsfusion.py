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


mupdates = 1000
data = np.loadtxt("twosensors.csv", delimiter=",")
testData = np.loadtxt("twosensors2.csv",delimiter=",")
inputs  = data[0:,0:11].astype(np.float32)
outputs = data[0:,11:12].astype(np.int32)
testInputs  = testData[0:,0:11].astype(np.float32)
testOutputs = testData[0:,11:12].astype(np.int32)

theautil.joint_shuffle(inputs,outputs)

train_and_valid, test = theautil.split_validation(90, inputs, outputs)
train, valid = theautil.split_validation(90,train_and_valid[0],train_and_valid[1])

def linit(x):
    return x.reshape((len(x),))

train = (train[0],linit(train[1]))
valid = (valid[0],linit(valid[1]))
test = (test[0],linit(test[1]))
testNew  = (testInputs ,linit(testOutputs))

#net = theanets.Classifier([11,(100,'softplus'),(500,'softplus'),(1000,'softplus'),(500,'softplus'),(250,'softplus'),(100,'softplus'),(2,'softplus')])
#net = theanets.Classifier([11,(100,'relu'),(500,'relu'),(1000,'relu'),(500,'relu'),(2,'relu')])
net = theanets.Classifier([11,50,500,500,200,50,2])
net.train(train, valid, algo='layerwise',momentum=0.9,max_updates=mupdates,hidden_dropout=0.5)
print "Finsh and Save model"
net.save("models/model([11,50,500,500,200,50,2]|algo=layerwise,momentum=0.9,max_updates=1000,hidden_dropout=0.5")
print "Learner on the test set"
classify = net.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])
print "Learner on the New test set"
classify = net.classify(testNew[0])
print "%s / %s " % (sum(classify == testNew[1]),len(testNew[1]))
print collections.Counter(classify)
print theautil.classifications(classify,testNew[1])



