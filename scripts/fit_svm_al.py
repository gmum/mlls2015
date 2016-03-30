#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Fit SVM in ActiveLearning experiment

 Note: if we use more models we should extract logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import optparse
import gzip
from os import path
import json
import cPickle

from six import iteritems

from models.cv import AdaptiveGridSearchCV
from models.balanced_models import RandomProjector, RandomNB
from training_data.datasets import CVBaseChemDataset
from bunch import Bunch
from experiments.utils import wac_score, wac_scoring
from misc.config import RESULTS_DIR
from misc.utils import config_log_to_file
from sklearn.svm import SVC
from misc.utils import get_run_properties
import alpy2
import alpy2
from alpy2.strategy import *
from alpy2.oracle import SimulatedOracle
from alpy2.utils import mask_unknowns
from sklearn.metrics import auc
from training_data.datasets import calculate_jaccard_kernel
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
import logging
import time

logger = logging.getLogger(__name__)

from alpy2.monitors import *
from alpy2.active import ActiveLearner
from alpy2.strategy import PassiveStrategy

def generate_time_report(monitor_outputs):
    # Returns dict with percentage/amount of time spent in each section (all keys with "_time" suffix)
    report = {}
    total_time = float(sum(monitor_outputs['iter_time']))
    for k in monitor_outputs:
        if k.endswith("_time"):
            # 1st: time in seconds, 2nd: percent of all time
            report[k] = [sum(monitor_outputs[k]), sum(monitor_outputs[k]) / total_time]
    return report


def calculate_scores(monitor_outputs):
    scores = {}
    for k in monitor_outputs:
        if "score" in k and isinstance(monitor_outputs[k], list) or isinstance(monitor_outputs[k], np.ndarray):
            if len(monitor_outputs[k]) == 0 or isinstance(monitor_outputs[k][0], list) or \
                    isinstance(monitor_outputs[k][0], np.ndarray):
                logger.info("Skipping calculation of scores for " + k + " because is a list of lists or is empty.")
                continue

            try:
                scores[k + "_mean"] = np.mean(monitor_outputs[k])
                scores[k + "_auc"] = auc(np.arange(len(monitor_outputs[k])), monitor_outputs[k])
            except Exception :
                logger.warning("Failed calculating score for " + k + ", might have been expected.")
    return scores


parser = optparse.OptionParser()
parser.add_option("--C_min", type="int", default=-6)
parser.add_option("--C_max", type="int", default=4)
parser.add_option("--holdout_cluster", type="string", default="", \
                  help="If non-empty value will be treated as meta key of clustering array and \
                        will be passed to get_meta. ")
parser.add_option("--internal_cv", type="int", default=3)
parser.add_option("--max_iter", type="int", default=50000000)
parser.add_option("--n_folds", type="int", default=5)
parser.add_option("--preprocess", type="str", default="max_abs")
parser.add_option("--fold", type="int", default=0)
parser.add_option("--d", type="int", default=1, help="AdaptiveGridSearchCV grid width")
parser.add_option("--output_dir", type="string", default=".")
parser.add_option("--warm_start", type="float", default=0.05)
parser.add_option("--strategy", type="string", default="PassiveStrategy")
parser.add_option("--strategy_kwargs", type="string", default="")
parser.add_option("--compound", type="str", default="5-HT1a")
parser.add_option("--representation", type="str", default="MACCS")
parser.add_option("--jaccard", type="int", default=1)
parser.add_option("--name", type="str", default="fit_svm_al")
parser.add_option("--rng", type="int", default=777)
parser.add_option("--batch_size", type="int", default=50)
parser.add_option("--model", type="str", default="SVM")

def _calculate_jaccard_kernel(X1T, X2T):
    X1T_sums = np.array(X1T.sum(axis=1))
    X2T_sums = np.array(X2T.sum(axis=1))
    K = X1T.dot(X2T.T)

    if hasattr(K, "toarray"):
        K = K.toarray()

    K2 = -(K.copy())
    K2 += (X1T_sums.reshape(-1, 1))
    K2 += (X2T_sums.reshape(1, -1))
    K = K / K2

    return K


def _kernel_to_distance_matrix(K):
    if not K.shape[0] == K.shape[1]:
        raise RuntimeError("Pass kernel matrix")

    D = np.sqrt(-2*K + K.diagonal().reshape(-1,1) + K.diagonal().reshape(1,-1))
    D /= D.max()
    return D


if __name__ == "__main__":

    start_time = time.time()
    (opts, args) = parser.parse_args()
    json_results = {}

    output_dir = opts.output_dir if path.isabs(opts.output_dir) else path.join(RESULTS_DIR, opts.output_dir)
    os.system("mkdir -p " + output_dir)

    config_log_to_file(fname=os.path.join(output_dir, opts.name + ".log"), clear_log_file=True)
    logger = logging.getLogger("fit_svm_al")

    logger.info(opts.__dict__)
    logger.info(opts.name)
    logger.info("Loading data..")

    duds =  "_DUDs" in opts.compound
    duds_train_mask = None
    duds_valid_mask = None
    duds_cluster_train_mask = None
    duds_cluster_valid_mask = None

    ### Prepare data ###

    data = CVBaseChemDataset(compound=opts.compound, representation=opts.representation, n_folds=opts.n_folds,
                             rng=opts.rng,
                             preprocess=opts.preprocess)

    (X_train, y_train), (X_valid, y_valid) = data.get_data(fold=opts.fold)
    X_train_cluster, X_valid_cluster = None, None

    if duds:
        duds_train_mask, duds_valid_mask = data.get_meta(fold=opts.fold, key="is_dud")

    if isinstance(opts.warm_start, float):
        assert opts.warm_start > 0 and opts.warm_start < 1
        warm_start_n = max(100, int(opts.warm_start * X_train.shape[0]))
    elif isinstance(opts.warm_start, int):
        warm_start_n = opts.warm_start
    else:
        raise TypeError("Wrong warm_start type")

    if len(opts.holdout_cluster):
        mask_train, mask_valid = data.get_meta(fold=opts.fold, key=opts.holdout_cluster)
        ids_train = np.where(mask_train==1)[0]
        ids_valid = np.where(mask_valid==1)[0]

        X_train_cluster, y_train_cluster = X_train[ids_train], y_train[ids_train]
        X_valid_cluster, y_valid_cluster = X_valid[ids_valid], y_valid[ids_valid]
        warm_start = np.random.RandomState(opts.rng).choice(np.where(mask_train==0)[0], size=warm_start_n, replace=False)

        if duds:
            duds_train_ids = np.where(duds_train_mask == 1)[0]
            duds_valid_ids = np.where(duds_valid_mask == 1)[0]

            duds_cluster_train_ids = np.hstack([np.where(ids_train == i)[0].tolist() for i in duds_train_ids]).astype("int")
            duds_cluster_valid_ids = np.hstack([np.where(ids_valid == i)[0].tolist() for i in duds_valid_ids]).astype("int")

            duds_cluster_train_mask = np.zeros(X_train_cluster.shape[0])
            duds_cluster_valid_mask = np.zeros(X_valid_cluster.shape[0])

            duds_cluster_train_mask[duds_cluster_train_ids] = 1
            duds_cluster_valid_mask[duds_cluster_valid_ids] = 1

        if opts.jaccard:
            assert opts.model != "RandomNB"
            logger.info("Calculating jaccard similarity between cluster and X_train")
            X_train_cluster, X_valid_cluster = \
                _calculate_jaccard_kernel(X_train_cluster, X_train), _calculate_jaccard_kernel(X_valid_cluster, X_train)

    else:
        warm_start = np.random.RandomState(opts.rng).choice(X_train.shape[0], size=warm_start_n, replace=False)

    json_results['warm_start'] = list(warm_start)

    # projection for CSJ
    if opts.strategy == "CSJSampling":
        random_projector = RandomProjector(rng=opts.rng)
        random_projector.fit(X_train)
        projection = random_projector.project(X_train)

    if opts.jaccard:
        assert opts.model != "RandomNB"
        logger.info("Calculating jaccard similarity between X_train and X_valid and X_train")
        X_train, X_valid = _calculate_jaccard_kernel(X_train, X_train), _calculate_jaccard_kernel(X_valid, X_train)
        pairwise_distance = _kernel_to_distance_matrix(X_train)
    elif opts.strategy in ["CSJSampling", "QuasiGreedyBatch"]:
        logger.info("Calculating jaccard similarity between X_train and X_valid and X_train")
        X_train, X_valid = _calculate_jaccard_kernel(X_train, X_train), _calculate_jaccard_kernel(X_valid, X_train)
        pairwise_distance = _kernel_to_distance_matrix(X_train)

    # Prepare y_train_masked
    warm_start = set(warm_start)
    y_train_masked = mask_unknowns(y_train, [i for i in range(y_train.shape[0]) if i not in warm_start])


    kernel = "precomputed" if opts.jaccard else "linear"

    try:
        strategy_kwargs = json.loads(opts.strategy_kwargs)
    except ValueError as e:
        raise ValueError("Cannot parse `strategy_kwargs` string, got: %s" % opts.strategy_kwargs)

    logger.info("Parsed strategy kwargs: " + str(strategy_kwargs))

    if opts.model == "SVM":
        if opts.d <= 0:
            estimator = GridSearchCV(
                estimator=SVC(random_state=opts.rng, kernel=kernel, max_iter=opts.max_iter, class_weight='balanced'),
                param_grid=
                {
                    "C": [10 ** c for c in range(opts.C_min, opts.C_max + 1)]},
                cv=opts.internal_cv,
                scoring=wac_scoring,
                error_score=0.)
        else:
            estimator = AdaptiveGridSearchCV(d=opts.d,
                                             estimator=SVC(random_state=opts.rng, kernel=kernel, max_iter=opts.max_iter,  class_weight='balanced'),
                                             param_grid=
                                             {
                                                 "C": [10 ** c for c in range(opts.C_min, opts.C_max + 1)]},
                                             cv=opts.internal_cv,
                                             scoring=wac_scoring,
                                             error_score=0.)
    elif opts.model == "RandomNB":

        assert opts.jaccard == 0
        NB_projector = RandomProjector()
        estimator = RandomNB(projector=NB_projector)

        if opts.strategy in ["CSJSampling", "QuasiGreedyBatch"]:
            strategy_kwargs['distance_cache'] = pairwise_distance
    elif opts.model == "NB":
        assert opts.jaccard == 0
        estimator = BernoulliNB(fit_prior=False)

        if opts.strategy in ["CSJSampling", "QuasiGreedyBatch"]:
            strategy_kwargs['distance_cache'] = pairwise_distance
    else:
        raise ValueError("Unknown model: %s" % opts.model)


    StrategyCls = getattr(alpy2.strategy, opts.strategy, getattr(alpy2.strategy, opts.strategy, None))
    if not StrategyCls:
        raise RuntimeError("Not found strategy " + opts.strategy)

    # cast non-string parametres
    for key, val in strategy_kwargs.iteritems():
        if key == "c":
            try:
                c = float(val)
            except ValueError as e:
                raise ValueError("Can't cast strategy parameter `c` to float, got {0}".format(val))
            strategy_kwargs[key] = c
        elif key in ["n_tries", "n_estimators"]:
            try:
                int_arg = int(val)
            except ValueError as e:
                raise ValueError("Can't cast strategy parameter `n_tries` to int, got {0}".format(val))
            strategy_kwargs[key] = int_arg

    if opts.strategy == "CSJSampling":
        strategy_kwargs['projection'] = projection


    strategy = StrategyCls(**strategy_kwargs)

    al = ActiveLearner(strategy=strategy,
                       random_state=opts.rng,
                       batch_size=opts.batch_size,
                       oracle=SimulatedOracle(sample_budget=np.inf),
                       estimator=estimator)

    monitors = []

    monitors.append(ExtendedMetricMonitor(name="wac_score",
                                          short_name="wac_score",
                                          function=wac_score,
                                          ids="all",
                                          frequency=1,
                                          duds_mask=duds_train_mask))

    monitors.append(ExtendedMetricMonitor(name="wac_score_labeled",
                                          short_name="wac_score_labeled",
                                          function=wac_score,
                                          ids="known",
                                          frequency=1,
                                          duds_mask=duds_train_mask))

    monitors.append(ExtendedMetricMonitor(name="wac_score_unlabeled",
                                          short_name="wac_score_unlabeled",
                                          function=wac_score,
                                          ids="unknown",
                                          frequency=1,
                                          duds_mask=duds_train_mask))

    monitors.append(ExtendedMetricMonitor(name="wac_score_valid",
                                          short_name="wac_score_valid",
                                          function=wac_score,
                                          frequency=1,
                                          X=X_valid,
                                          y=y_valid,
                                          duds_mask=duds_valid_mask))

    if len(opts.holdout_cluster):
        monitors.append(ExtendedMetricMonitor(name="wac_score_valid_aleph",
                                              short_name="wac_score_valid_aleph",
                                              function=wac_score,
                                              frequency=1,
                                              X=X_valid_cluster,
                                              y=y_valid_cluster,
                                              duds_mask=duds_cluster_valid_mask))

        monitors.append(ExtendedMetricMonitor(name="wac_score_train_aleph",
                                              short_name="wac_score_train_aleph",
                                              function=wac_score,
                                              frequency=1,
                                              X=X_train_cluster,
                                              y=y_train_cluster,
                                              duds_mask=duds_cluster_train_mask))

    monitors.append(SimpleLogger(batch_size=opts.batch_size, frequency=10))

    monitors.append(EstimatorMonitor(only_params=True))

    if isinstance(estimator, GridSearchCV):
        monitors.append(GridScoresMonitor())

    al.fit(X_train, y_train_masked, monitors=monitors)


    # Save results

    json_results['cmd'] = "{} {}".format(__file__,
                                         " ".join("--{} {}".format(k, v) for k, v in iteritems(opts.__dict__)))
    json_results['opts'] = opts.__dict__
    json_results['dataset'] = data.get_params()
    json_results['time_reports'] = generate_time_report(al.monitor_outputs_)
    json_results['scores'] = calculate_scores(al.monitor_outputs_)
    json_results['run'] = get_run_properties()

    json_results['time_reports']['total_time'] = time.time() - start_time

    json.dump(json_results, open(path.join(output_dir, opts.name + ".json"), "w"), indent=4, sort_keys=True)
    cPickle.dump(al.monitor_outputs_, gzip.open(path.join(output_dir, opts.name + ".pkl.gz"), "w"))
