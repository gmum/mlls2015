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
from dataset.utils import update_meta
from sklearn.datasets import dump_svmlight_file

sabina_files = glob.glob(path.join(DATA_DIR, "sabina", "*.csv"))



X_all = {}
y_all = {}
X_duds = {}
y_duds = {}
for sabina_file in sabina_files:
    print sabina_file
    # Skip first row and first column
    X = pd.read_csv(sabina_file, sep=",")
    X = X.drop("Name", 1)
    X = X.as_matrix()
    y = np.array([int("_actives" in sabina_file) for _ in xrange(X.shape[0])])
    target_file = path.basename(sabina_file)
    target_file = target_file.replace("5ht1a", "5-HT1a")
    target_file = target_file.replace("5ht2a", "5-HT2a")
    target_file = target_file.replace("5ht7", "5-HT7")
    target_file = target_file.replace("5ht6", "5-HT6")
    target_file = target_file.replace("5ht2c", "5-HT2c")
    target_file = target_file.replace("_actives", "")
    target_file = target_file.replace("_inactives", "")
    target_file = target_file.replace("_DUDs", "")

    if "DUDs" in path.basename(sabina_file):

        if X.shape[0] > 4200:
            ids = np.random.RandomState(777).choice(range(X.shape[0]), 4200, replace=False)
            X = X[ids]
            y = y[ids]
            print "Subsampled %s" % sabina_file

        X_duds[target_file] = X
        y_duds[target_file] = y
    else:
        if target_file in X_all:
            X_all[target_file] = np.vstack([X_all[target_file], X])
            y_all[target_file] = np.hstack([y_all[target_file], y])
        else:
            X_all[target_file] = X
            y_all[target_file] = y

print "Discovered keys", sorted(X_all.keys())

for key in X_all:
    print "Saving to " + key.split("_")[1][0:-6]
    fname = key.replace(".csv", ".libsvm")
    dirname = key.split("_")[1][0:-6]
    os.system("mkdir -p " + os.path.join(DATA_DIR, dirname))

    # Remove NaN rows
    nan_rows = set(np.where(np.isnan(X_all[key]))[0])
    print "Found " + str(nan_rows) + " nans"
    X_all[key] = X_all[key][[i for i in range(X_all[key].shape[0]) if i not in nan_rows]]
    y_all[key] = y_all[key][[i for i in range(y_all[key].shape[0]) if i not in nan_rows]]
    assert X_all[key].shape[0] == y_all[key].shape[0]

    if key in X_duds:
        nan_rows_duds = set(np.where(np.isnan(X_duds[key]))[0])
        print "Found " + str(nan_rows_duds) + " nans in DuD subset"

        X_duds[key] = X_duds[key][[i for i in range(X_duds[key].shape[0]) if i not in nan_rows_duds]]
        y_duds[key] = y_duds[key][[i for i in range(y_duds[key].shape[0]) if i not in nan_rows_duds]]

        assert key in X_duds, "All proteins should have DuDs chosen"
        X_with_duds = np.vstack([X_all[key], X_duds[key]])
        y_with_duds = np.hstack([y_all[key], y_duds[key]])
        is_dud = [0] * len(y_all[key]) + [1] * len(y_duds[key])
        assert X_with_duds.shape[0] == y_with_duds.shape[0], "Samples size is consistent"
        print "Writing: ", os.path.join(DATA_DIR, dirname, fname.split("_")[0] + "_DUDs_" + fname.split("_")[1])
        target = os.path.join(DATA_DIR, dirname, fname.split("_")[0] + "_DUDs_" +
                                                                    fname.split("_")[1])
        dump_svmlight_file(X_with_duds, y_with_duds, f=target, zero_based=True)
        update_meta(target[0:-7] + ".meta", {"is_dud": np.array(is_dud).astype("int")})
    else:
        print "Warning. Missing DUDs for", key


    print "Writing: ", os.path.join(DATA_DIR, dirname, fname)
    dump_svmlight_file(X_all[key], y_all[key], f=os.path.join(DATA_DIR, dirname, fname), zero_based=True)
