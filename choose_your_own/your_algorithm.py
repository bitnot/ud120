#!/usr/bin/python

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing


import sys

import matplotlib.pyplot as plt

sys.path.insert(0, '../lib')
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually

def plot_xy(X, Y, marker='o', alpha=0.5):
    grade_fast = [X[ii][0] for ii in range(0, len(X)) if Y[ii] == 0]
    bumpy_fast = [X[ii][1] for ii in range(0, len(X)) if Y[ii] == 0]
    grade_slow = [X[ii][0] for ii in range(0, len(X)) if Y[ii] == 1]
    bumpy_slow = [X[ii][1] for ii in range(0, len(X)) if Y[ii] == 1]
    # initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast", marker=marker, alpha=alpha)
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow", marker=marker, alpha=alpha)

plot_xy(features_train, labels_train, marker='x', alpha=0.5)
plot_xy(features_test, labels_test, marker='o', alpha=1)

plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.title("training data")
plt.show()
plt.savefig("initial.png")
output_image('initial.png')

################################################################################

print(f'samples train = {len(features_train)}, test = {len(features_test)}')
print(f'fast % train ={sum(labels_train)*100.0/len(labels_train)}, test ={sum(labels_test)*100.0/len(labels_test)}')

# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary

classifiers = [
    (DecisionTreeClassifier(min_samples_leaf=8, random_state=0), {
        # 'criterion': ['gini', 'entropy'],
        # 'splitter': ['random', 'best'],
        'min_samples_split': [2, 4, 8, 16, 32],
        'max_depth': [None, 4, 8, 10, 12, 16, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_leaf_nodes': [None, 20, 40, 100, 200]
    }),
    (RandomForestClassifier(random_state=0, n_jobs=-1), {
        # 'criterion': ['gini', 'entropy'],
        # 'min_weight_fraction_leaf': [0.0, 0.01, 0.1],
        # 'min_impurity_decrease': [0.0, 0.01, 0.1],
        # 'warm_start': [False, True],
        # 'class_weight' : ["balanced", "balanced_subsample"],
        # 'ccp_alpha': [0.0],
        # 'max_features' : ["auto", "sqrt", "log2"],
        # 'min_samples_split': [2, 4, 8, 16, 32],
        # 'max_depth': [None, 1, 2, 3, 4, 5, 8, 10, 12, 16, 20],
        # 'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_leaf_nodes': [None, 20, 40, 100, 200],
        'n_estimators': [20, 30, 50, 100],
        # 'n_estimators': [1, 2, 4, 8, 20, 50, 100],
        # 'max_leaf_nodes': [None, 20, 40, 100, 200]
    }),
    (KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, n_jobs=-1, n_neighbors=15, p=1), {
        'n_neighbors': [3, 5, 15, 20, 25, 50],
        'weights': ['uniform', 'distance'],
        # ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [10, 20, 30, 40],
        'p': [1, 2, 4],
        'metric': ['minkowski']
    }),
    (AdaBoostClassifier(learning_rate=0.7, n_estimators=20, random_state=0), {
        'base_estimator': [ None ],
        'n_estimators': [20, 50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0],
        'algorithm': ['SAMME.R', 'SAMME'],
    }),
    (SVC(kernel="rbf", C=10.0, random_state=0), {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        # 'degree': [3],
        # 'gamma': [1.0],
        # 'coef0': [0.0],
        # 'shrinking': [True],
        # 'probability': [False],
        'C':[1.0, 5.0, 9.0, 10.0, 15.0, 20, 100],
        # 'tol': [1e-3],
        # 'cache_size': [200],
        # 'class_weight': [None],
        # 'max_iter': [-1],
        # 'decision_function_shape':['ovo', 'ovr'],
        # 'break_ties': [False]
    }),
    (GaussianNB(priors=[0.5, 0.5]),{
        'var_smoothing': [1e-9, 1e-3, 1e-12],
        'priors':[None, [0.5, 0.5], [0.35, 0.65], [0.1, 0.9], [0.9, 0.1]]
    })
]



results = []

for base_clf, param_grid in classifiers:
    try:
        search = GridSearchCV(base_clf, param_grid, n_jobs=-1)
        search.fit(features_train, labels_train)
        score = search.best_estimator_.score(features_test, labels_test)
        result = {
            'score': score,
            'clf': search.best_estimator_,
            'params': search.best_params_
        }
        name = str(search.best_estimator_).split('(')[0]
        title = f'{name} {score:5.4f}'
        prettyPicture(search.best_estimator_, features_test,
                    labels_test, name, title)
        output_image(f'{name}.png')
        print(result)
        results.append(result)
    except Exception as e:
        print(e)

best = max(results, key=lambda x: x['score'])
print(f'best result: {best}')

"""
samples train = 750, test = 250
fast % train =64.13333333333334, test =66.4
{'score': 0.924, 'clf': DecisionTreeClassifier(min_samples_leaf=8, random_state=0), 'params': {'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 8, 'min_samples_split': 2}}
{'score': 0.92, 'clf': RandomForestClassifier(max_leaf_nodes=40, n_estimators=20, n_jobs=-1,
                       random_state=0), 'params': {'max_leaf_nodes': 40, 'n_estimators': 20}}
{'score': 0.928, 'clf': KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, n_jobs=-1,
                     n_neighbors=15, p=1), 'params': {'algorithm': 'ball_tree', 'leaf_size': 10, 'metric': 'minkowski', 'n_neighbors': 15, 'p': 1, 'weights': 'uniform'}}
{'score': 0.928, 'clf': AdaBoostClassifier(learning_rate=0.5, n_estimators=20, random_state=0), 'params': {'algorithm': 'SAMME.R', 'base_estimator': None, 'learning_rate': 0.5, 'n_estimators': 20}}
{'score': 0.94, 'clf': SVC(C=10.0, random_state=0), 'params': {'C': 10.0, 'kernel': 'rbf'}}
{'score': 0.904, 'clf': GaussianNB(priors=[0.5, 0.5]), 'params': {'priors': [0.5, 0.5], 'var_smoothing': 1e-09}}
best result: {'score': 0.94, 'clf': SVC(C=10.0, random_state=0), 'params': {'C': 10.0, 'kernel': 'rbf'}}
"""
