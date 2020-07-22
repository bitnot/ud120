import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from lib.class_vis import prettyPicture
from lib.prep_terrain_data import makeTerrainData
from classify import NBAccuracy


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

if __name__ == "__main__":
    print(f"Accuracy is {submitAccuracy()}")
