#!/usr/bin/python

import sys

import matplotlib.pyplot as plt

sys.path.insert(0, '../lib')
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.titile("training data")
plt.show()
plt.savefig("initial.png")
output_image('initial.png')

################################################################################

print(f'samples train = {len(features_train)}, test = {len(features_test)}')

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

classifiers = [
    (DecisionTreeClassifier(min_samples_leaf=8, random_state=0), {
    # 'criterion': ['gini', 'entropy'],
    # 'splitter': ['random', 'best'],
    'min_samples_split': [2, 4, 8, 16, 32],
    'max_depth': [None, 4, 8, 10, 12, 16, 20],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_leaf_nodes': [None, 20, 40, 100, 200]
    }),
    (RandomForestClassifier(max_leaf_nodes=40, n_estimators=20, random_state=0, n_jobs=-1), {
    # 'criterion': ['gini', 'entropy'],
    # 'splitter': ['random', 'best'],
    # 'min_samples_split': [2, 4, 8, 16, 32],
    # 'max_depth': [None, 4, 8, 10, 12, 16, 20],
    # 'min_samples_leaf': [1, 2, 4, 8, 16],
    # 'max_leaf_nodes': [None, 20, 40, 100, 200],
    'n_estimators': [2, 4, 8, 20, 50, 100],
    'max_leaf_nodes': [None, 20, 40, 100, 200]
    }),
    (KNeighborsClassifier(n_jobs=-1, n_neighbors=15, p=1), {
        'n_neighbors': [3, 5, 15, 20, 25, 50],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'], # ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40],
        'p': [1, 2, 4],
        'metric': ['minkowski']
    }),
    (AdaBoostClassifier(learning_rate=0.7, n_estimators=20, random_state=0),{
        'base_estimator': [None],
        'n_estimators': [8, 15, 20, 25, 30, 40, 50],
        'learning_rate': list(reversed([0.1, 0.4, 0.5, 0.7, 0.8, 1.0])),
        'algorithm': ['SAMME.R', 'SAMME'],
    })
]

results = []

for base_clf, param_grid in classifiers:
    search = GridSearchCV(base_clf, param_grid, n_jobs=-1)
    search.fit(features_train, labels_train)
    score = search.best_estimator_.score(features_test, labels_test)
    result = {
        'clf': search.best_estimator_,
        'score': score,
        'params': search.best_params_
    }
    name = str(search.best_estimator_).split('(')[0]
    title = f'{name} {score:0.4f}'
    prettyPicture(search.best_estimator_, features_test, labels_test, name, title)
    output_image(f'{name}.png')
    print(result)
    results.append(result)

best = max(results, key=lambda x: x['score'])
print(f'best result: {best}')

"""
samples train = 750, test = 250
{'clf': DecisionTreeClassifier(min_samples_leaf=8, random_state=0), 'score': 0.924, 'params': {'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 8, 'min_samples_split': 2}}
{'clf': RandomForestClassifier(max_leaf_nodes=40, n_estimators=20, n_jobs=-1, random_state=0), 'score': 0.92, 'params': {'max_leaf_nodes': 40, 'n_estimators': 20}}
{'clf': KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, n_jobs=-1, n_neighbors=15, p=1), 'score': 0.928, 'params': {'algorithm': 'ball_tree', 'leaf_size': 10, 'metric': 'minkowski', 'n_neighbors': 15, 'p': 1, 'weights': 'uniform'}}
{'clf': AdaBoostClassifier(learning_rate=0.7, n_estimators=20, random_state=0), 'score': 0.928, 'params': {'algorithm': 'SAMME.R', 'base_estimator': None, 'learning_rate': 0.7, 'n_estimators': 20}}

best result: {  'clf': KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, n_jobs=-1, n_neighbors=15, p=1),
                'score': 0.928,
                'params': {'algorithm': 'ball_tree', 'leaf_size': 10, 'metric': 'minkowski', 'n_neighbors': 15, 'p': 1, 'weights': 'uniform'}}
"""
