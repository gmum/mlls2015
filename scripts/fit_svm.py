#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM to given compound.

 Note: it doesn't use active learning evaluation
"""
import json
import logging
from os import path
import matplotlib.pylab as plt
from misc.config import RESULTS_DIR, LOG_DIR
from misc.utils import config_log_to_file, utc_date
from experiments.utils import dict_hash, wac_score
from training_data.datasets import *
from training_data.datasets import calculate_jaccard_kernel
from sklearn.svm import SVC, LinearSVC
from six import iteritems
import optparse
import cPickle
import gzip

parser = optparse.OptionParser()
parser.add_option("--C_min", type="int", default=-6)
parser.add_option("--C_max", type="int", default=4)
parser.add_option("--n_folds", type="int", default=5)
parser.add_option("--preprocess", type="str", default="max_abs")
parser.add_option("--fold", type="int", default=1)
parser.add_option("--compound", type="str", default="beta2")
parser.add_option("--representation", type="str", default="SRMACCS")
parser.add_option("--output_dir", type="str", default="fit_svm_SRMACCS")
parser.add_option("--name", type="str", default="")
parser.add_option("--jaccard", type="int", default=1)
parser.add_option("--rng", type="int", default=777)
(opts, args) = parser.parse_args()

if __name__ == "__main__":
    output_dir = opts.output_dir if os.path.isabs(opts.output_dir) else path.join(RESULTS_DIR, opts.output_dir)
    os.system("mkdir -p " + output_dir)
    name = opts.name if opts.name else "fit_svm_{0}".format(dict_hash(opts.__dict__))

    config_log_to_file(os.path.join(output_dir, name + ".log"), clear_log_file=True)
    logger = logging.getLogger("fit_svm")
    logger.info(opts.__dict__)
    logger.info(name)

    # Get data
    data = CVBaseChemDataset(compound=opts.compound, representation=opts.representation, n_folds=opts.n_folds, rng=opts.rng,
                           preprocess=opts.preprocess)
    (X_train, y_train), (X_valid, y_valid) = data.get_data(fold=opts.fold)
    if opts.jaccard:
        K_train, K_valid = calculate_jaccard_kernel(data=data, fold=opts.fold)

    # Calculate results
    results = {}
    C_range = range(opts.C_min, opts.C_max+1)
    grid = [{"C": 10**i} for i in C_range]
    for params in grid:
        logger.info("Testing " + str(params))
        if opts.jaccard:
            m = SVC(kernel="precomputed", class_weight="balanced",random_state=opts.rng,  **params)
            m.fit(K_train, y_train)
            y_pred = m.predict(K_valid)
            results[str(params)] = {"y_pred": y_pred, "wac": wac_score(y_valid, y_pred), "clf": m}
        else:
            m = LinearSVC(loss="hinge",class_weight="balanced", random_state=opts.rng, **params)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_valid)
            results[str(params)] = {"y_pred": y_pred, "wac": wac_score(y_valid, y_pred), "clf": m}

    wac_scores = [results[str(params)]["wac"] for params in grid]
    logger.info("Max WAC=" + str(np.max(wac_scores)))
    best_results = results[str(grid[np.argmax(wac_scores)])]

    # Save results
    json_results = dict(best_results)
    json_results['cmd'] = "{} {}".format(__file__, " ".join("--{} {}".format(k, v) for k, v in iteritems(opts.__dict__)))
    json_results['opts'] = opts.__dict__
    json_results['grid'] = grid
    json_results['wac_scores'] = wac_scores
    json_results['dataset'] = data.get_params()
    json_results['clf'] = json_results['clf'].get_params()
    del json_results['y_pred']

    json.dump(json_results, open(path.join(output_dir, name + ".json"), "w"), indent=4, sort_keys=True)
    cPickle.dump(results, gzip.open(path.join(output_dir, name + ".pkl.gz"), "w"))