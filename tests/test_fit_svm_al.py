# -*- coding: utf-8 -*-
"""
 Tests for fit_svm_al. Checks reproducibility,
 final AUC score meeting point, uncertainty sampling
 superiority over passive strategy and final score
 being close to fitting separately SVM
"""

import os
import sys
from os import path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training_data.datasets import CVBaseChemDataset
from bunch import Bunch

import json
from experiments.utils import wac_score, wac_scoring

from misc.config import BASE_DIR, RESULTS_DIR
import gzip

import cPickle
from sklearn.svm import SVC

from training_data.datasets import calculate_jaccard_kernel

from sklearn.grid_search import GridSearchCV


def test_fit_svm_al():
    opts_uncert = Bunch({"C_min": -6,
                         "C_max": 5,
                         "internal_cv": 4,
                         "max_iter": 10000000,
                         "n_folds": 5,
                         "preprocess": "clip01",  # "max_abs",
                         "fold": 3,
                         "d": 1,
                         "output_dir": "test_fit_svm_al",
                         "warm_start": 20,  # TODO: add cluster-dependent warm_start
                         "strategy_kwargs": r"{}",
                         "strategy": "UncertaintySampling",
                         "compound": "beta2",
                         "representation": "MACCS",
                         "jaccard": 1,
                         "rng": 777,
                         "name": "uncertainty",
                         "batch_size": 50})

    opts_uncert_2 = Bunch(opts_uncert)
    opts_uncert_2['name'] = 'uncertainty_2'

    opts_passive = Bunch({"C_min": -6,
                          "C_max": 5,
                          "internal_cv": 4,
                          "max_iter": 10000000,
                          "n_folds": 5,
                          "preprocess": "clip01",  # "max_abs",
                          "fold": 3,
                          "d": 1,
                          "output_dir": "test_fit_svm_al",
                          "warm_start": 20,  # TODO: add cluster-dependent warm_start
                          "strategy_kwargs": r"{}",
                          "strategy": "PassiveStrategy",
                          "compound": "beta2",
                          "representation": "MACCS",
                          "jaccard": 1,
                          "rng": 777,
                          "name": "passive",
                          "batch_size": 50})

    opts_passive_2 = Bunch(opts_passive)
    opts_passive_2['name'] = 'passive_2'

    jobs = [opts_uncert, opts_uncert_2, opts_passive, opts_passive_2]

    # Run jobs
    for job in jobs:
        cmd = "./scripts/fit_svm_al.py " + " ".join("--{} {}".format(k, v) for k, v in job.iteritems() if v)
        cmd = path.join(BASE_DIR, cmd)
        print "Running ", cmd
        print os.system("cd ..;" + cmd)

    # Load results and compare/plot
    p1 = json.load(open(path.join(RESULTS_DIR, "test_fit_svm_al/passive.json")))
    p2 = json.load(open(path.join(RESULTS_DIR, "test_fit_svm_al/passive_2.json")))
    u1 = json.load(open(path.join(RESULTS_DIR, "test_fit_svm_al/uncertainty.json")))
    u2 = json.load(open(path.join(RESULTS_DIR, "test_fit_svm_al/uncertainty_2.json")))

    # Scores are replicable given rng
    for k in u1['scores']:
        if "time" not in k:
            assert u1['scores'][k] == u2['scores'][k], k + " should be replicable"

    for k in p1['scores']:
        if "time" not in k:
            assert p1['scores'][k] == p2['scores'][k], k + " should be replicable"

    p1_mon = cPickle.load(gzip.open(path.join(RESULTS_DIR, "test_fit_svm_al/passive.pkl.gz")))
    u1_mon = cPickle.load(gzip.open(path.join(RESULTS_DIR, "test_fit_svm_al/uncertainty.pkl.gz")))

    # Converge to same score
    for k in u1_mon:
        if "score" in k and "time" not in k:
            assert p1_mon[k][-1] == u1_mon[k][-1], "Last score for " + k + " should be the same"

    assert u1['scores']['wac_score_valid_auc'] > p1['scores']['wac_score_valid_auc']

    # Check that last WAC score result
    opts = jobs[0]

    data = CVBaseChemDataset(compound=opts.compound, representation=opts.representation,
                             n_folds=opts.n_folds, rng=opts.rng,
                             preprocess=opts.preprocess)
    (X_train, y_train), (X_valid, y_valid) = data.get_data(fold=opts.fold)
    if opts.jaccard:
        K_train, K_valid = calculate_jaccard_kernel(data=data, fold=opts.fold)

    C_range = range(opts.C_min, opts.C_max + 1)
    param_grid = {"C": [10 ** i for i in C_range]}
    m = GridSearchCV(param_grid=param_grid, estimator=
    SVC(kernel="precomputed", class_weight="balanced", random_state=opts.rng))
    m.fit(K_train, y_train)

    assert abs(u1_mon['wac_score_valid'][-1] - wac_score(y_valid, m.predict(K_valid))) < 0.01
