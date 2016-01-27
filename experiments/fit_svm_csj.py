#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM (Jaccard/not) with query by bagging
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

config_log_to_file(fname=os.path.join(LOG_DIR,  "fit_svm_query_by_bagging.log"), clear_log_file=True)
logger = logging.getLogger("fit_svm_query_by_bagging")

N_FOLDS = 5

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=10)

def _get_job_opts(jaccard, fold, strategy, batch_size, csj_c, fp):
    opts = {"C_min": -6,
            "C_max": 5,
            "internal_cv": 3,
            "max_iter": 8000000,
            "n_folds": N_FOLDS,
            "preprocess": "max_abs",
            "fold": fold,
            "d": 1,
            "warm_start": 20,
            "strategy_kwargs": r'{\"c\":\"' + str(csj_c) + r'\"}',
            "strategy": strategy,
            "compound": "5-HT1a",
            "representation": fp,
            "jaccard": jaccard,
            "rng": 777,
            "batch_size": batch_size,
            "holdout_cluster": "validation_clustering"}

    opts['name'] = dict_hash(opts)
    opts['output_dir'] = path.join(RESULTS_DIR, fp, "SVM-csj-"+str(csj_c))
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
    for csj_c in [0.4, 0.5, 0.6, 0.7]:
        for fp in ['Klek', 'Ext']:
            for batch_size in [20, 50, 100]:
                for f in range(N_FOLDS):
                    for j in [1]: # jaccard = 0 is super slow!
                        jobs.append(["./scripts/fit_svm_al.py", _get_job_opts(jaccard=j,
                                                                              strategy="CSJSampling",
                                                                              batch_size=batch_size,
                                                                              fold=f,
                                                                              csj_c=csj_c,
                                                                              fp=fp)])

            run_async_with_reporting(run_job, jobs, n_jobs=opts.n_jobs, output_dir=path.join(RESULTS_DIR, fp,"SVM-csj-"+str(csj_c)))