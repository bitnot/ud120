#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.insert(0,'../../lib')

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV


features_train, labels_train, features_test, labels_test = makeTerrainData()

tree=DecisionTreeClassifier(
                #  criterion="gini",
                #  splitter="best",
                #  max_depth=None,
                #  min_samples_split=2,
                #  min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 #  random_state=None,
                 random_state=0,
                #  max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort='deprecated',
                 ccp_alpha=0.0)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['random', 'best'],
    'min_samples_split': [2, 4, 8, 16, 32, 50, 100],
    'max_depth': [None, 2, 4, 10, 50, 30, 100],
    'min_samples_leaf': [1, 2, 4, 10, 20, 50, 100],
    'max_leaf_nodes': [None, 2, 4, 10, 20, 50, 100]
    }
search = GridSearchCV(tree, param_grid, n_jobs=4, cv=5)
search.fit(features_train, labels_train)

print(search.best_params_)
# print(search.best_score_)
clf = search.best_estimator_

acc = clf.score(features_test, labels_test)

def submitAccuracy():
    return acc

print(acc)

#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image('test.png')
