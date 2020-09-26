"""
Lesson 3 Quiz 12
"""
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sys
sys.path.insert(0,'../../lib')

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
# we handle the import statement and SVC creation for you here
clf = SVC(kernel="linear") #, gamma=1.0, C=1.0


# now your job is to fit the classifier
# using the training features/labels, and to
# make a set of predictions on the test data
clf.fit(features_train, labels_train)

# store your predictions in a list named pred
pred = clf.predict(features_test)


acc = accuracy_score(pred, labels_test)


def submitAccuracy():
    return acc

print(acc)

prettyPicture(clf, features_test, labels_test)
output_image('test.png')
