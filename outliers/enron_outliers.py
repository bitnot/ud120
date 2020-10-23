#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import mpld3

sys.path.append("../ud120-projects/tools/")
from feature_format import targetFeatureSplit


def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    import numpy as np
    return_list = []
    return_keys = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = list(dictionary.keys())

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )
            return_keys.append(key)

    return np.array(return_list), return_keys


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../ud120-projects/final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data, keys = featureFormat(data_dict, features)

outlier = max(data.tolist())
print(outlier)
outlier_key = [key for key in data_dict
    if [data_dict[key]["salary"], data_dict[key]["bonus"]] == outlier ][0]
print(outlier_key)
data_dict.pop( outlier_key, 0 )
data, keys = featureFormat(data_dict, features)


### your code below
salary, bonus = zip(*data.tolist())

fig, ax = plt.subplots()

scatter = ax.scatter( salary, bonus )

ax.set_xlabel("salary")
ax.set_ylabel("bonus")

tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=keys)
mpld3.plugins.connect(fig, tooltip)

# ax.show()
mpld3.show()

