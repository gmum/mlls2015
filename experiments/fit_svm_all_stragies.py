#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM (Jaccard) with all previously used strategies
"""

import numpy as np
import optparse
from experiments.utils import run_async_with_reporting, dict_hash, run_job
from os import path
from misc.config import RESULTS_DIR, LOG_DIR
from misc.utils import config_log_to_file
import logging
import cPickle, gzip
import os

config_log_to_file(fname=os.path.join(LOG_DIR,  "fit_svm_all_strategies.log"), clear_log_file=True)
logger = logging.getLogger("fit_svm_all_strategies")

N_FOLDS = 5

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=10)

def _get_job_opts(jaccard, fold, strategy, batch_size):
    opts = {"C_min": -6,
                  "C_max": 5,
                  "internal_cv": 3,
                  "max_iter": 8000000,
                  "n_folds": N_FOLDS,
                  "preprocess": "max_abs",
                  "fold": fold,
                  "d": 1,
                  "output_dir": path.join(RESULTS_DIR, "SVM-all"),
                  "warm_start": 20,
                  "strategy_kwargs": r"{}",
                  "strategy": strategy,
                  "compound": "5-HT1a",
                  "representation": "MACCS",
                  "jaccard": jaccard,
                  "rng": 777,
                  "batch_size": batch_size}

    opts['name'] = dict_hash(opts)
    return opts

def get_results(jaccard, strategy, batch_size):
    # Load all monitors
    results = []
    for f in range(N_FOLDS):
        args = _get_job_opts(jaccard=jaccard, batch_size=batch_size, strategy=strategy, fold=f)
        monitors_file = path.join(args['output_dir'], dict_hash(args) + ".pkl.gz")
        if path.exists(monitors_file):
            results.append(cPickle.load(gzip.open(monitors_file)))

if __name__ == "__main__":
    (opts, args) = parser.parse_args()
    jobs = []
    for strategy in ['UncertaintySampling', 'PassiveStrategy', 'QuasiGreedyBatch']:
        for batch_size in [20, 50, 100]:
            for f in range(N_FOLDS):
                for j in [1]: # jaccard = 0 is super slow!
                    jobs.append(["./scripts/fit_svm_al.py", _get_job_opts(jaccard=j, strategy=strategy, batch_size=batch_size, fold=f)])

    run_async_with_reporting(run_job, jobs, n_jobs=opts.n_jobs, output_dir=path.join(RESULTS_DIR, "SVM-all"))