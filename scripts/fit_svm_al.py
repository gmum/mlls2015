#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM in ActiveLearning experiment

 Note: if we use more models we should extract logic
"""

import optparse
import gzip
from os import path
import os
import json
import cPickle

from six import iteritems

from models.cv import AdaptiveGridSearchCV
from training_data.datasets import CVBaseChemDataset
from bunch import Bunch
from experiments.utils import wac_score
from misc.config import RESULTS_DIR
from misc.utils import config_log_to_file
from sklearn.svm import SVC
from misc.utils import get_run_properties
import alpy
import alpy_addons
from alpy.strategy import *
from alpy.oracle import SimulatedOracle
from alpy.utils import mask_unknowns
from sklearn.metrics import auc
from training_data.datasets import calculate_jaccard_kernel
from sklearn.grid_search import GridSearchCV


from alpy_addons.monitors import *
from alpy_addons.active import ActiveLearner
from alpy_addons.strategy import PassiveStrategy

def generate_time_report(monitor_outputs):
    # Returns dict with percentage/amount of time spent in each section (all keys with "_time" suffix)
    report = {}
    total_time = float(sum(monitor_outputs['iter_time']))
    for k in monitor_outputs:
        if k.endswith("_time"):
            report[k] = [sum(monitor_outputs[k]), sum(monitor_outputs[k]) / total_time]
    return report


def calculate_scores(monitor_outputs):
    scores = {}
    for k in monitor_outputs:
        if "score" in k:
            scores[k + "_mean"] = np.mean(monitor_outputs[k])
            scores[k + "_auc"] = auc(np.arange(len(monitor_outputs[k])), monitor_outputs[k])
    return scores

def wac_scoring(estimator, X, y):
    return wac_score(y, estimator.predict(X))

parser = optparse.OptionParser()
parser.add_option("--C_min", type="int", default=-6)
parser.add_option("--C_max", type="int", default=4)
parser.add_option("--internal_cv", type="int", default=2)
parser.add_option("--max_iter", type="int", default=50000000)
parser.add_option("--n_folds", type="int", default=5)
parser.add_option("--preprocess", type="str", default="max_abs")
parser.add_option("--fold", type="int", default=0)
parser.add_option("--d", type="int", default=1, help="AdaptiveGridSearchCV grid width")
parser.add_option("--output_dir", type="string", default=".")
parser.add_option("--warm_start", type="int", default=50)
parser.add_option("--strategy", type="string", default="PassiveStrategy")
parser.add_option("--strategy_kwargs", type="string", default="")
parser.add_option("--compound", type="str", default="5-HT1a")
parser.add_option("--representation", type="str", default="MACCS")
parser.add_option("--jaccard", type="int", default=1)
parser.add_option("--name", type="str", default="")
parser.add_option("--rng", type="int", default=777)
parser.add_option("--batch_size", type="int", default=50)
(opts, args) = parser.parse_args()

if __name__ == "__main__":
    output_dir = opts.output_dir if path.isabs(opts.output_dir) else path.join(RESULTS_DIR, opts.output_dir)
    os.system("mkdir -p " + output_dir)

    config_log_to_file(fname=os.path.join(output_dir, opts.name + ".log"), clear_log_file=True)
    logger = logging.getLogger("fit_svm")
    logger.info(opts.__dict__)
    logger.info(opts.name)

    logger.info("Loading data..")
    data = CVBaseChemDataset(compound=opts.compound, representation=opts.representation, n_folds=opts.n_folds,
                             rng=opts.rng,
                             preprocess=opts.preprocess)
    (X_train, y_train), (X_valid, y_valid) = data.get_data(fold=opts.fold)
    if opts.jaccard:
        X_train, X_valid = calculate_jaccard_kernel(data=data, fold=opts.fold)

    y_train_masked = mask_unknowns(y_train,
                                   np.random.choice(X_train.shape[0],
                                                    size=X_train.shape[0] - opts.warm_start, replace=False))

    kernel = "precomputed" if opts.jaccard else "linear"

    if opts.d <= 0:
        estimator = GridSearchCV(
            estimator=SVC(random_state=opts.rng, kernel=kernel, max_iter=opts.max_iter),
            param_grid=
            {
                "C": [10 ** c for c in range(opts.C_min, opts.C_max + 1)]},
            cv=opts.internal_cv,
            scoring=wac_scoring,
            error_score=0.)
    else:
        estimator = AdaptiveGridSearchCV(d=opts.d,
                                         estimator=SVC(random_state=opts.rng, kernel=kernel, max_iter=opts.max_iter),
                                         param_grid=
                                         {
                                             "C": [10 ** c for c in range(opts.C_min, opts.C_max + 1)]},
                                         cv=opts.internal_cv,
                                         scoring=wac_scoring,
                                         error_score=0.)

    StrategyCls = getattr(alpy_addons.strategy, opts.strategy, getattr(alpy.strategy, opts.strategy, None))
    if not StrategyCls:
        raise RuntimeError("Not found strategy " + opts.strategy)
    strategy_kwargs = {token.split()[0]: float(token.split()[1]) for token in opts.strategy_kwargs.split()}
    logger.info("Parsed strategy kwargs: " + str(strategy_kwargs))
    strategy = StrategyCls(**strategy_kwargs)

    al = ActiveLearner(strategy=strategy,
                       random_state=opts.rng,
                       batch_size=opts.batch_size,
                       oracle=SimulatedOracle(sample_budget=np.inf),
                       estimator=estimator)

    # TODO: add cluster monitors
    monitors = []

    monitors.append(ExtendedMetricMonitor(name="wac_score",
                                          short_name="wac_score",
                                          function=wac_score,
                                          ids="all",
                                          frequency=1))

    monitors.append(ExtendedMetricMonitor(name="wac_score_labeled",
                                          short_name="wac_score_labeled",
                                          function=wac_score,
                                          ids="known",
                                          frequency=1))

    monitors.append(ExtendedMetricMonitor(name="wac_score_unlabeled",
                                          short_name="wac_score_unlabeled",
                                          function=wac_score,
                                          ids="unknown",
                                          frequency=1))

    monitors.append(ExtendedMetricMonitor(name="wac_score_valid",
                                          short_name="wac_score_valid",
                                          function=wac_score,
                                          frequency=1,
                                          X=X_valid,
                                          y=y_valid))

    monitors.append(SimpleLogger(batch_size=opts.batch_size))

    monitors.append(EstimatorMonitor(only_params=True))

    monitors.append(GridScoresMonitor())

    al.fit(X_train, y_train_masked, monitors=monitors)


    # Save results
    json_results = {}
    json_results['cmd'] = "{} {}".format(__file__,
                                         " ".join("--{} {}".format(k, v) for k, v in iteritems(opts.__dict__)))
    json_results['opts'] = opts.__dict__
    json_results['dataset'] = data.get_params()
    json_results['time_reports'] = generate_time_report(al.monitor_outputs_)
    json_results['scores'] = calculate_scores(al.monitor_outputs_)
    json_results['run'] = get_run_properties()

    json.dump(json_results, open(path.join(output_dir, opts.name + ".json"), "w"), indent=4, sort_keys=True)
    cPickle.dump(al.monitor_outputs_, gzip.open(path.join(output_dir, opts.name + ".pkl.gz"), "w"))
