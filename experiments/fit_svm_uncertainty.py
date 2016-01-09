#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM (Jaccard/not) with uncertainty strategy
"""

import numpy as np
import matplotlib as plt
import optparse
from .utils import run_async_with_reporting, dict_hash
from os import path
from misc.config import RESULTS_DIR
import cPickle, gzip

N_FOLDS = 5

def opts(jaccard, fold):
  return {"C_min": -6,
                  "C_max": 5,
                  "internal_cv": 4,
                  "max_iter": 50000000,
                  "n_folds": 5,
                  "preprocess": "max_abs",
                  "fold": fold,
                  "d": 1,
                  "output_dir": path.join(RESULTS_DIR, "SVM-uncertainty"),
                  "warm_start": 20,
                  "strategy_kwargs": "",
                  "strategy": "UncertaintySampling",
                  "compound": "5-HT1a",
                  "representation": "MACCS",
                  "jaccard": jaccard,
                  "rng": 777,
                  "batch_size": 50}

def get_results(jaccard):
    # Load all monitors
    results = []
    for f in range(N_FOLDS):
        args = opts(jaccard=jaccard)
        monitors_file = path.join(args['output_dir'], dict_hash(args) + ".pkl.gz")
        if path.exists(monitors_file):
            results.append(cPickle.load(gzip.open(monitors_file)))

if __name__ == "__main__":
    jobs = []
    for f in range(N_FOLDS):
        for j in [0]:
            jobs.append(opts(jaccard=j, fold=f))

    run_async_with_reporting(jobs, n_jobs=2)