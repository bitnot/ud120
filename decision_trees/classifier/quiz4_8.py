#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.insert(0,'../../lib')

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify
from sklearn.metrics import accuracy_score


features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)


# store your predictions in a list named pred
# pred = clf.predict(features_test)
# acc = accuracy_score(pred, labels_test)
acc = clf.score(features_test, labels_test)

def submitAccuracy():
    return acc

print(acc)

#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image('test.png')
