#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../../ud120-projects/tools/")
from email_preprocess import preprocess

TOOLS_PATH = "../../ud120-projects/tools"
words_file = f"{TOOLS_PATH}/word_data.pkl"
authors_file = f"{TOOLS_PATH}/email_authors.pkl"

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess(
    words_file, authors_file)


#########################################################
### your code goes here ###


#########################################################

def timeit(code, label=""):
    t0 = time()
    result = code()
    print(f"time {label}: {round(time()-t0, 3)}s")
    return result


def make_clf(features_train, labels_train, sample=1.0, params={}):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0, **params)

    # sample training set to 1%
    if sample < 1.0:
        features_train = features_train[:int(len(features_train)*sample)]
        labels_train = labels_train[:int(len(labels_train)*sample)]

    # fit the classifier on the training features and labels
    timeit(lambda: clf.fit(features_train, labels_train), "fit")

    return clf


def get_accuracy(clf, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    from sklearn.metrics import accuracy_score

    # use the trained classifier to predict labels for the test features
    labels_pred = timeit(lambda: clf.predict(features_test), "predict")

    # calculate and return the accuracy on the test data
    accuracy = accuracy_score(labels_test, labels_pred)
    return accuracy

print(f'number of features: {len(features_train[0])}')

# sampled_clf = make_clf(features_train, labels_train, sample=0.1, params={'min_samples_split': 40})
# print(f"sampled_clf accuracy is {get_accuracy(sampled_clf, features_test, labels_test):.4f}")

clf = make_clf(features_train, labels_train, params={'min_samples_split': 40})
print(f"clf accuracy is {get_accuracy(clf, features_test, labels_test):.4f}")

# percentile=10 number of features: 3785
# time fit: 49.547s
# time predict: 0.031s
# clf accuracy is 0.9790

# percentile=1  number of features:  379
# time fit: 3.284s
# time predict: 0.003s
# clf accuracy is 0.9670

# rbf_sampled_c10_clf = make_clf(features_train, labels_train, sample=0.1, kernel="linear", C=10)
# print(f"rbf_sampled_c10_clf accuracy is {get_accuracy(rbf_sampled_c10_clf, features_test, labels_test)}")

# rbf_sampled_c100_clf = make_clf(features_train, labels_train, sample=0.1, kernel="linear", C=100)
# print(f"rbf_sampled_c100_clf accuracy is {get_accuracy(rbf_sampled_c100_clf, features_test, labels_test)}")

# rbf_sampled_c1000_clf = make_clf(features_train, labels_train, sample=0.1, kernel="linear", C=1000)
# print(f"rbf_sampled_c1000_clf accuracy is {get_accuracy(rbf_sampled_c1000_clf, features_test, labels_test)}")

# rbf_sampled_c10000_clf = make_clf(features_train, labels_train, sample=0.1, kernel="linear", C=10000)
# print(f"rbf_sampled_c10000_clf accuracy is {get_accuracy(rbf_sampled_c10000_clf, features_test, labels_test)}")

# rbf_c10000_clf = make_clf(features_train, labels_train, sample=1.0, kernel="linear", C=10000)
# print(f"rbf_c10000_clf accuracy is {get_accuracy(rbf_c10000_clf, features_test, labels_test)}")

# preds = rbf_sampled_c10000_clf.predict([features_test[10],features_test[26],features_test[50]])
# print(f"rbf_sampled_c10000_clf preditions: {preds}")
