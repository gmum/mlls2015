#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM (Jaccard/not) with quasi greedy batch
"""

import numpy as np
import optparse
from experiments.utils import run_async_with_reporting, dict_hash, run_job, get_output_dir
from os import path
from misc.config import RESULTS_DIR, LOG_DIR
from misc.utils import config_log_to_file
import logging
import cPickle, gzip
import os
from alpy2.strategy import QGB_DIST_AVG, QGB_DIST_GLOBAL_MIN

config_log_to_file(fname=os.path.join(LOG_DIR,  "fit_svm_quasi_greedy.log"), clear_log_file=True)
logger = logging.getLogger("fit_svm_quasi_greedy")

N_FOLDS = 5

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=10)


def _get_job_opts(jaccard, fold, model, compound, strategy, batch_size, qgb_c, fingerprint, qgb_dist):

    output_dir = get_output_dir(model, compound, fingerprint, strategy, param=qgb_c, special="QGB_SUB")

    opts = {"C_min": -6,
            "C_max": 5,
            "internal_cv": 3,
            "max_iter": 8000000,
            "n_folds": N_FOLDS,
            "preprocess": "max_abs",
            "fold": fold,
            "d": 1,
            "warm_start": 0.05,
            "strategy_kwargs": r'{"c":"' + str(qgb_c) + r'"' + r',' + r'"dist_fnc":"' + str(qgb_dist) + r'"' + r',' + r'"n_tries":"10"}',
            "strategy": strategy,
            "compound": compound,
            "representation": fingerprint,
            "jaccard": jaccard,
            "rng": 777,
            "batch_size": batch_size,
            "holdout_cluster": "validation_clustering"}

    opts['name'] = dict_hash(opts)
    opts["output_dir"] = output_dir
    opts['model'] = model
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
    model = "SVM"
    strategy = 'QuasiGreedyBatch'
    for compound in ["5-HT2c", "5-HT2a", "5-HT6", "5-HT7", "5-HT1a", "d2"]:
        for qgb_c in [0.9, 0.99, 0.999, 0.1, 0.3]:
            for qgb_dist in [QGB_DIST_GLOBAL_MIN, QGB_DIST_AVG]:
                for fingerprint in ['Ext', 'Klek', 'Pubchem']:
                    for batch_size in [20, 50, 100]:
                        for f in range(N_FOLDS):
                            for j in [1]: # jaccard = 0 is super slow!
                                jobs.append(["./scripts/fit_svm_al.py", _get_job_opts(jaccard=j,
                                                                                      strategy=strategy,
                                                                                      batch_size=batch_size,
                                                                                      fold=f,
                                                                                      compound=compound,
                                                                                      model=model,
                                                                                      fingerprint=fingerprint,
                                                                                      qgb_c=qgb_c,
                                                                                      qgb_dist=qgb_dist)])

                    output_dir = get_output_dir(model, compound, fingerprint, strategy, param=qgb_c, special="QGB_SUB")
                    run_async_with_reporting(run_job, jobs, n_jobs=opts.n_jobs, output_dir=output_dir)
