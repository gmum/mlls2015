#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM (Jaccard/not) with uncertainty strategy
"""

import numpy as np
import optparse
from experiments.utils import run_async_with_reporting, dict_hash, run_job
from os import path
from misc.config import RESULTS_DIR
import cPickle, gzip

N_FOLDS = 5

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=10)

def _get_job_opts(jaccard, fold, batch_size):
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
                  "batch_size": batch_size}

def get_results(jaccard, batch_size):
    # Load all monitors
    results = []
    for f in range(N_FOLDS):
        args = _get_job_opts(jaccard=jaccard, batch_size=batch_size, fold=f)
        monitors_file = path.join(args['output_dir'], dict_hash(args) + ".pkl.gz")
        if path.exists(monitors_file):
            results.append(cPickle.load(gzip.open(monitors_file)))

if __name__ == "__main__":
    (opts, args) = parser.parse_args()
    jobs = []
    for batch_size in [20, 50, 100]:
        for f in range(N_FOLDS):
            for j in [0]:
                jobs.append(["./scripts/fit_svm_al.py", _get_job_opts(jaccard=j, batch_size=batch_size, fold=f)])

    run_async_with_reporting(run_job, jobs, n_jobs=opts.n_jobs)