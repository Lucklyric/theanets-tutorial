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

net = theanets.Classifier([11,(100,'softmax'),(500,'softmax'),(1000,'softmax'),(500,'softmax'),(250,'softmax'),(100,'softmax'),(2,'softmax')])
#net = theanets.Classifier([11,(500,'sigmoid'),(1000,'sigmoid'),(500,'sigmoid'),(2,'sigmoid')])
net.train(train, valid, algo='layerwise', max_updates=mupdates, patience=1,save_every=100,save_progress="model{}")
print "Finsh and Save model"
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



