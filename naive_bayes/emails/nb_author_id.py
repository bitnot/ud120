#!/usr/bin/env python3

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels

TOOLS_PATH = "../../ud120-projects/tools"
words_file = f"{TOOLS_PATH}/word_data.pkl"
authors_file = f"{TOOLS_PATH}/email_authors.pkl"
features_train, features_test, labels_train, labels_test = preprocess(
    words_file, authors_file)


#########################################################
### your code goes here ###

#########################################################
