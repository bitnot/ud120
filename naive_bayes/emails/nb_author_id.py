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
def timeit(code, label=""):
    t0 = time()
    result = code()
    print(f"time {label}: {round(time()-t0, 3)}s")
    return result

### your code goes here ###


def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    # create classifier
    clf = GaussianNB()

    # fit the classifier on the training features and labels
    timeit(lambda: clf.fit(features_train, labels_train), "fit")

    # use the trained classifier to predict labels for the test features
    labels_pred = timeit(lambda: clf.predict(features_test), "predict")

    # calculate and return the accuracy on the test data
    # this is slightly different than the example,
    # where we just print the accuracy
    # you might need to import an sklearn module
    accuracy = accuracy_score(labels_test, labels_pred)
    return accuracy


print(
    f"Accuracy is {NBAccuracy(features_train, labels_train, features_test, labels_test)}")
#########################################################
