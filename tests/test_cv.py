# -*- coding: utf-8 -*-
"""
 Simple tests for models in models.cv
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV
from models.cv import AdaptiveGridSearchCV
from training_data.datasets import CVBaseChemDataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from bunch import Bunch

def wac_score(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)
    if cm.shape != (2,2):
        return accuracy_score(Y_true, Y_pred)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    if tp == 0 and fn == 0:
        return 0.5*tn/float(tn+fp)
    elif tn == 0 and fp == 0:
        return 0.5*tp/float(tp+fn)
    return 0.5*tp/float(tp+fn) + 0.5*tn/float(tn+fp)

def wac_scoring(estimator, X, y):
    return wac_score(y, estimator.predict(X))

def test_AdaptiveGridSearchCV():
    opts = Bunch({"C_min": -7,
              "C_max": 6,
              "max_iter": 80000000,
              "n_folds": 5,
              "preprocess": "clip01", #"max_abs",
              "fold": 2,
              "compound": "beta2",
              "representation": "MACCS",
              "jaccard": 0,
              "rng": 777,
              "name": "test_svm_al",
              "batch_size": 100,
              "output_dir": "/Users/kudkudak/code/mlls2015/"})

    m1 = GridSearchCV(
                           estimator=SVC(random_state=opts.rng, max_iter=opts.max_iter,  class_weight='balanced'),
                           param_grid =
                               {
                                 "C": [5**c for c in range(opts.C_min, opts.C_max + 1)]},
                           cv=3,
                           scoring=wac_scoring,
                           error_score=0.)

    m2 = AdaptiveGridSearchCV(d=1,
                           estimator=SVC(random_state=opts.rng,  max_iter=opts.max_iter,  class_weight='balanced'),
                           param_grid =
                               {
                                 "C": [5**c for c in range(opts.C_min, opts.C_max + 1)]},
                           cv=3,
                           scoring=wac_scoring,
                           error_score=0.)

    data = CVBaseChemDataset(compound="SERT", representation=opts.representation, n_folds=opts.n_folds, rng=opts.rng,
                           preprocess=opts.preprocess)
    (X_train, y_train), _ = data.get_data(fold=opts.fold)
    data = CVBaseChemDataset(compound="beta2", representation=opts.representation, n_folds=opts.n_folds, rng=opts.rng,
                           preprocess=opts.preprocess)
    (X_train2, y_train2), _ = data.get_data(fold=opts.fold)


    m1.fit(X_train[0:100], y_train[0:100])
    opt_C_before = m1.best_params_['C']
    m1.fit(X_train2[0:100], y_train2[0:100])
    opt_C_after = m1.best_params_['C']

    assert opt_C_before / opt_C_after > 5 or opt_C_before / opt_C_after < 1/5., "Test should be actually testing \
        optimal parameters differing by more than AdaptiveGridSearchCV.d"

    m2.fit(X_train[0:100], y_train[0:100])
    opt_C_before = m2.best_params_['C']
    m2.fit(X_train2[0:100], y_train2[0:100])
    opt_C_after =  m2.best_params_['C']

    assert opt_C_before / opt_C_after <= 5 or opt_C_before / opt_C_after >= 1/5., \
        "AdaptiveGridSearchCV shouldn't look further than d"

    m2.fit(X_train[0:100], y_train[0:100])
    opt_C_before_2 = m2.best_params_['C']
    m2.fit(X_train2[0:100], y_train2[0:100])
    opt_C_after_2 =  m2.best_params_['C']

    assert opt_C_before == opt_C_before_2 and opt_C_after == opt_C_after_2, "Results should be repetitive"