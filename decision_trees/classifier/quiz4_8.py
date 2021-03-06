#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.insert(0,'../../lib')

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn.metrics import accuracy_score


features_train, labels_train, features_test, labels_test = makeTerrainData()

def classify(features_train, labels_train):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 #  random_state=None,
                 random_state=0,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort='deprecated',
                 ccp_alpha=0.0)
    clf.fit(features_train, labels_train, sample_weight=None, check_input=True, X_idx_sorted=None)
    return clf

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
