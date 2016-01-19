#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Converts all raw data to libsvm format

 (Expects sabina folder with .csv files in DATA_DIR)
"""
import os
from os import path
import glob
import numpy as np
from misc.config import DATA_DIR
import pandas as pd
from sklearn.datasets import dump_svmlight_file

sabina_files = glob.glob(path.join(DATA_DIR, "sabina", "*.csv"))

X_all = {}
y_all = {}
for sabina_file in sabina_files:
    print sabina_file
    # Skip first row and first column
    X = pd.read_csv(sabina_file, sep=",")
    X = X.drop("Name", 1)
    X = X.as_matrix()
    y = np.array([int("_actives" in sabina_file) for _ in xrange(X.shape[0])])
    sabina_file = path.basename(sabina_file)
    sabina_file = sabina_file.replace("5ht1a", "5-HT1a")
    sabina_file = sabina_file.replace("_actives", "")
    sabina_file = sabina_file.replace("_inactives", "")

    if sabina_file in X_all:
        X_all[sabina_file] = np.vstack([X_all[sabina_file], X])
        y_all[sabina_file] = np.hstack([y_all[sabina_file], y])
    else:
        X_all[sabina_file] = X
        y_all[sabina_file] = y

for key in X_all:
    print "Saving to " + key.split("_")[1][0:-6]
    fname = key.replace(".csv", ".libsvm")
    dirname = key.split("_")[1][0:-6]
    print "Found " + str(np.isnan(X_all[key]).sum()) + " nans"
    X_all[key][np.isnan(X_all[key])] = 0
    os.system("mkdir -p " + os.path.join(DATA_DIR, dirname))
    assert X_all[key].shape[0] == y_all[key].shape[0]
    dump_svmlight_file(X_all[key], y_all[key], f=os.path.join(DATA_DIR, dirname, fname), zero_based=True)
