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


testData = np.loadtxt("twosensors2.csv",delimiter=",")
testInputs  = testData[0:,0:11].astype(np.float32)
testOutputs = testData[0:,11:12].astype(np.int32)


def linit(x):
    return x.reshape((len(x),))

testNew  = (testInputs ,linit(testOutputs))

#net.load(sys.argv[1])
net= theanets.Classifier.load('models/model([11,20,2]|algo=layerwise,momentum=0.9,max_updates=1000)')

print "Learner on the New test set"
classify = net.classify(testNew[0])
print "%s / %s " % (sum(classify == testNew[1]),len(testNew[1]))
print collections.Counter(classify)
print theautil.classifications(classify,testNew[1])

