import json
import cPickle
import gzip

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import traceback

from collections import defaultdict, OrderedDict
from itertools import product

from misc.config import RESULTS_DIR, CACHE_DIR
from experiments.utils import dict_hash


logger = logging.getLogger(__name__)


### stuff for analyzing results ###

def load_json(file_path):
    """
    Loads json file to python dict
    :param file_path: string, path ot file
    :return: dict, python dictionary with json contents
    """
    with open(file_path, 'r') as f:
        content = json.load(f)
    return content


def load_pklgz(file_path):
    """
    Load .pkl.gz file from file to python dict
    :param file_path: string, path ot file
    :return: dict, python dictionary with file contents
    """
    with gzip.open(file_path, 'r') as f:
        content = cPickle.load(f)
    return content


def get_mean_experiments_results(results_dir, strategies, batch_sizes=[20, 50, 100]):
    """
    Read all experiments from given directory, calculated mean results for all combinations of
    strategies and batch_sizes
    :param results_dir: string, path to directiory with results
    :param batch_sizes: list of ints, batch sizes the experiments were run on
    :param strategies: string, strategies the experimtens were run on, default 'all'
    :return: dict of dicts: {strategy-batchsize: {metric: mean_results}}
    """

    assert isinstance(strategies, str)

    if strategies == 'all':
        strategies = ['UncertaintySampling', 'PassiveStrategy', 'QueryByBagging']
    else:
        assert results_dir[-3:] == strategies[-3:]
        strategies = [strategies]

    mean_scores = {strategy + '-' + str(batch_size): defaultdict(list) for strategy in strategies for batch_size in batch_sizes}

    for results_file in filter(lambda x: x[-4:] == 'json', os.listdir(results_dir)):

        json_path = os.path.join(results_dir, results_file)
        assert os.path.exists(json_path)

        json_results = load_json(json_path)
        strategy = json_results['opts']['strategy']
        batch_size = json_results['opts']['batch_size']

        if strategy in ["CSJSampling", "QuasiGreedyBatch"]:
            strategy_kwargs = json.loads(json_results['opts']['strategy_kwargs'])
            param_c = strategy_kwargs['c']
            key = strategy + "-"  + str(param_c) + '-' + str(batch_size)
        else:
            key = strategy + '-' + str(batch_size)

        assert key in mean_scores

        for metric, score in json_results['scores'].iteritems():
            mean_scores[key][metric].append(score)

        pkl_path = os.path.join(results_dir, results_file[:-5] + ".pkl.gz")
        assert os.path.exists(pkl_path)

        scores = load_pklgz(pkl_path)

        for metric, values in scores.iteritems():
            mean_scores[key][metric].append(values)

    for strategy, scores in mean_scores.iteritems():
        for metric, values in scores.iteritems():
            if '_mon' in metric \
                    or '_predictions' in metric \
                    or '_true' in metric \
                    or 'selected_' in metric:
                continue
            try:
                if '_auc' in metric or '_mean' in metric:
                    mean_scores[strategy][metric] = np.mean(values)
                else:
                    mean_scores[strategy][metric] = np.vstack(values).mean(axis=0)
            except Exception as e:
                traceback.format_exc(e)
                import pdb
                pdb.set_trace()

            if '_auc' in metric or '_mean' in metric:
                assert isinstance(mean_scores[strategy][metric], float)
            else:
                assert mean_scores[strategy][metric].shape[0] > 1

    return mean_scores


def pick_best_param_c_experiment(results_dir, strategy, metric):

    assert strategy in ["CSJSampling", "QuasiGreedyBatch"]
    assert "_auc" in metric or "_mean" in metric

    if strategy == 'CSJSampling':
        short_strat = 'csj'
    elif strategy == 'QuasiGreedyBatch':
        short_strat = 'qgb'

    result_dirs = [os.path.join(results_dir, 'SVM-' + short_strat + '-' + str(c)) for c in [0.3, 0.4, 0.5, 0.6, 0.7]]

    best_result = {str(bs): ("", 0) for bs in [20, 50, 100]}
    for res_dir in result_dirs:
        mean_res = get_mean_experiments_results(res_dir, strategies=strategy + res_dir[-4:])
        for strat, scores in mean_res.iteritems():
            batch_size = strat.split('-')[-1]
            if metric not in scores.keys():
                raise ValueError("Worng metric: %s" % metric)

            score = scores[metric]
            assert isinstance(score, float)
            if score > best_result[batch_size][1]:
                best_result[batch_size] = (strat, scores)

    ret = {}
    for bs, (strat, scores) in best_result.iteritems():
        ret[strat] = scores
    return ret

def compare_curves(scores, metrics=['wac_score_valid'], batch_sizes=[20, 50, 100]):
    """
    Plot curves from given mean scores and metrics
    :param scores: dict of dicts, mean scores for every combination of params
    :param metrics: list of strings or string, which metrics to plot
    :param batch_sizes: list of ints, batch sizes the experiments were run on
    :return:
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    fig, axes = plt.subplots(len(metrics) * len(batch_sizes), 1)
    fig.set_figwidth(15)
    fig.set_figheight(8 * len(axes))

    exp_type = product(batch_sizes, metrics)

    scores = OrderedDict(sorted(scores.items()))

    # plot all passed metrics
    for ax, (batch_size, metric) in zip(axes, exp_type):
        for strategy, score in scores.iteritems():
            if strategy.split('-')[-1] == str(batch_size):
                strategy_name = "-".join(strategy.split('-')[:-1])
                d = {strategy_name: score[metric]}
                pd.DataFrame({strategy_name: score[metric]}).plot(title='%s %d batch size' % (metric, batch_size), ax=ax)
                ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5))


def plot_curves(results_dir, metrics, best_param_metric, cached=True):
    """
    Plots curves for mean results off all experiments in given directory for given metrics
    :param results_dir: string, path to results directory
    :param metrics: list of strings or string, which metrics to plot
    :return:
    """
    name = dict_hash({'path': results_dir, 'best_param_metric': best_param_metric})
    cache_file = os.path.join(CACHE_DIR, "experiments.analyze", name + ".pkl.gz")
    if cached and os.path.exists(cache_file):
        print("Loading cache...")
        mean_scores = load_pklgz(cache_file)
    else:
        print("Processing results...")
        # `all` strategies
        mean_scores = {}
        all_dir = os.path.join(results_dir, "SVM-all")
        all_scores = get_mean_experiments_results(all_dir, strategies='all')
        mean_scores.update(all_scores)

        # csj and qgb
        for strategy in ["CSJSampling", "QuasiGreedyBatch"]:
            best_strat_res = pick_best_param_c_experiment(results_dir, strategy, metric=best_param_metric)
            for key in best_strat_res.keys():
                assert key not in mean_scores.keys()
            mean_scores.update(best_strat_res)

        if not os.path.exists(cache_file):
            with gzip.open(cache_file, 'w') as f:
                cPickle.dump(mean_scores, f)

    compare_curves(mean_scores, metrics=metrics)


def get_results_per_strategy(results_dir, strategies='all'):
    """
    Returns a python dict with results from single experiment per strategy, gets first experiment found.
    :param results_dir: string, directory with saved results
    :param strategies: 'all' or list of strings, for which strategies to return results
    :return: dict
    """

    assert isinstance(strategies, str) or isinstance(strategies, list)

    if strategies == 'all':
        strategies = ['UncertaintySampling',
                      'PassiveStrategy',
                      'QuasiGreedyBatch',
                      'QueryByBagging',
                      'CSJSampling']
    else:
        for strategy in strategies:
            assert strategy in ['UncertaintySampling', 'PassiveStrategy', 'QuasiGreedyBatch', 'QueryByBagging',
                                'CSJSampling']

    strategy_results = {strategy: None for strategy in strategies}

    for json_file in filter(lambda x: x[-4:] == "json", os.listdir(results_dir)):
        json_results = load_json(os.path.join(results_dir, json_file))

        strategy = json_results['opts']['strategy']
        if strategy not in strategies:
            continue
        if strategy_results[strategy] is None:
            strategy_results[strategy] = json_results

        if None not in strategy_results.values():
            break

    assert None not in strategy_results.values()
    return strategy_results


def get_time_reports_per_strategy(results_dir, strategies='all'):
    """
    Extracts time reports from first expermient per strategy found in given directory
    :param results_dir: string, directory with saved results
    :param strategies: strategies: 'all' or list of strings, for which strategies to return results
    :return: dict
    """

    strategy_results = get_results_per_strategy(results_dir, strategies)

    time_reports= {}
    for strategy, results in strategy_results.iteritems():
        time_reports[strategy] = results['time_reports']

    return time_reports


def plot_pie_chart(time_reports):
    """
    Plots pie charts for every time report dict given
    :param time_reports: dict, strategy_name: time_report_dict
    :return: None
    """

    n = len(time_reports.keys())
    fig = plt.figure(figsize=(11, 7 * n ))

    for i, (strategy_name, time_report) in enumerate(time_reports.iteritems()):

        labels = ['other']
        sizes = [0.]
        other = []

        for key, values in time_report.iteritems():
            if key not in ["total_time", "iter_time"]:
                if values[1] < 0.02:
                    sizes[0] += values[0]
                    other.append(key)
                else:
                    labels.append(key)
                    sizes.append(values[0])

        explode = [0.03] * len(labels)

        sorted_data = sorted(zip(labels, sizes), key=lambda x: x[1])

        sizes = [d[1] for d in sorted_data]
        labels = [d[0] for d in sorted_data]

        ax = fig.add_subplot(n, 1, i + 1)
        cs = plt.cm.Set1(np.arange(10) / 10.)
        patches, _, _ = ax.pie(sizes, colors=cs, shadow=True, explode=explode, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(strategy_name)
        ax.legend(patches, labels, loc='best')

    fig.tight_layout()


def plot_time_pie_charts(results_dir, strategies='all'):
    """
    Plots pie charts for time reports for first found results file per strategy
    :param results_dir: string, directory with saved results
    :param strategies: strategies: 'all' or list of strings, for which strategies to return results
    :return: None
    """

    time_reports = get_time_reports_per_strategy(results_dir, strategies=strategies)
    plot_pie_chart(time_reports)


def get_all_pickle_results(results_dir):
    """
    Loads all results in .pkl.gz files in given directory
    :param results_dir: string, directory with saved results
    :return: dict, file_path: results_dict
    """

    pickle_results = {}

    for pickle_file in filter(lambda x: 'pkl.gz' in x, os.listdir(results_dir)):
        results = load_pklgz(os.path.join(results_dir, pickle_file))

        pickle_results[pickle_file] = results

    return pickle_results


def check_extreme_params_all_results():
    """
    Checks if AdaptiveGridSearchCV picked an extreme parameter as best in WHOLE results directory
    :return: None
    """
    logger.warning("This may take at least a minute!")
    for results_dir in filter(lambda x: 'SVM' in x, os.listdir(RESULTS_DIR)):
        check_extreme_params(os.path.join(RESULTS_DIR, results_dir))


def check_extreme_params(results_dir):
    """
    Checks if AdaptiveGridSearchCV picked an extreme parameter as best in given results directory
    :param results_dir: string, directory with saved results
    :return: None
    """

    all_results = get_all_pickle_results(results_dir)
    found = False
    for i, pickle_results in enumerate(all_results):
        if not _single_check_for_extreme_params:
            logger.warning("Extreme params in %s" % pickle_results.keys())
            found = True

    if found:
        logger.warning( "FOUND SOME PICKED EXTREME PARAMETERS in %s" % results_dir)
    else:
        logger.info("No extreme params found in %s :)" % results_dir)


def _single_check_for_extreme_params(pickle_results, min_c=1e-6, max_c=1e5):
    """
    Check for extreme parameters in given results
    :param pickle_results: dict, experiment results
    :param min_c: float, minimum parameter value
    :param max_c: float, maximum parameter value
    :return: True if there were no extreme parameters picked, False if there were
    """

    best_params = []

    for params in pickle_results['grid_mon']:
        best_params.append(max(params, key=lambda x: x[1])[0].values())

    best_cs = np.unique(best_params)
    return min_c not in best_cs and max_c not in best_cs


def get_corresponding_experiment_results(exp_file, compare_results_dir):
    """
    Returns results of the experiment with the same name (same run parameters) in given directory
    :param exp_file: string, path to base experiment results
    :param compare_results_dir: string, directory in which to look for corresponding experiment
    :return: dict, results of corresponding experiment
    """

    assert exp_file[-4:] == "json"

    base_file_name = exp_file.split("/")[-1]
    file_path = os.path.join(compare_results_dir, base_file_name)
    assert os.path.exists(file_path)
    results = load_json(file_path)

    return results


def compare_json_results(exp_file, compare_results_dir):
    """
    Find and compare results of an experiment with the same parameters as given
    :param exp_file: string, path to base experiment results
    :param string, directory in which to look for corresponding experiment
    :return: None
    """

    base_results = load_json(exp_file)
    compare_results = get_corresponding_experiment_results(exp_file, compare_results_dir)

    assert base_results.keys() == compare_results.keys()

    # results will differ in:
    #    * run results - PID, slight code modification, etc.
    #    * time_reports - depending on machine load
    #    * scores - ONLY IN TIMES! Yes, there are times saved there
    for key in base_results.keys():
        if base_results[key] != compare_results[key]:
            assert key in ['run', 'time_reports', 'scores']

    # make sure differences in scores are only in times
    for (k1, s1), (k2, s2) in zip(base_results['scores'].iteritems(), compare_results['scores'].iteritems()):
        if abs(s1 - s2) > 0:
            assert 'time' in k1