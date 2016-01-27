# -*- coding: utf-8 -*-
"""
 Various utils functions for running experiments
"""
from os import path, system
import os
from six import iteritems
import logging
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
import time
from multiprocessing import Pool
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import gzip
from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import pandas as pd
from misc.config import RESULTS_DIR
import pdb

logger = logging.getLogger(__name__)

def upload_df_to_drive(df, name="test.csv"):
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)
    f = drive.CreateFile({'title':name, 'mimeType':'text/csv', "parents":
        [{"kind": "drive#fileLink", "id": "0B9cVNObhE5w9RzNFOEt1VFFmRUk"}]})
    f.SetContentString(df.to_csv())
    f.Upload(param={'convert': True})



def _check_duplicates(tasks):
    """ Checks that name is unique """
    tasks_dict = {}
    for t in tasks:
        if t[1]['name'] in tasks:
            logger.error(tasks_dict[1]['name'])
            logger.error(t[1])
            raise RuntimeError("Duplicated name in tasks")
        tasks_dict[t[1]['name']] = t

    already_calculated = 0
    for name in tasks_dict:

        kwargs = tasks_dict[name][1]


        target = path.join(kwargs['output_dir'], name) + ".json"
        if path.exists(target):
            already_calculated += 1
            done_job = json.load(open(target))
            shared_items = set(kwargs.items()) & set(done_job['opts'].items())
            if not (len(shared_items) == len(kwargs) == len(done_job['opts'])):
                print set(kwargs.items()) - set(done_job['opts'].items())
                raise RuntimeError("Found calculated job with same name but different parameters in %s" % target)

    if already_calculated:
        logger.warning("Skipping calculation of " + str(already_calculated) + " jobs (already calculated)")



def run_async_with_reporting(f, tasks, output_dir, n_jobs):


    # Naming should be unique and dir shouldn't have duplicated jobs already calculated
    _check_duplicates(tasks)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(path.join(output_dir, "failed_jobs.err"), "w") as _:
        pass

    with open(path.join(output_dir, "duplicated_jobs.err"), "w") as _:
        pass

    elapsed = 0
    burn_in_time = 9
    started_with = 0

    rs = Pool(n_jobs).map_async(f, tasks, chunksize=1)

    while True :
        if rs.ready():
            logger.info("Done")
            break
        remaining = rs._number_left
        logger.info(("Waiting for", remaining, "tasks to complete"))

        time.sleep(3)
        elapsed += 3.0
        if elapsed > burn_in_time:
            if started_with == 0:
                started_with = remaining
            completed = started_with - remaining
            if completed > 0:
                logger.info(("Estimated time is: ", (remaining * (elapsed - burn_in_time)) / float(completed)))

    if os.stat(path.join(output_dir, "duplicated_jobs.err")).st_size != 0:
        raise RuntimeError("Some jobs were duplicated")

    if os.stat(path.join(output_dir, "failed_jobs.err")).st_size != 0:
        raise RuntimeError("Some jobs failed")

    return rs.get()

def dict_hash(my_dict):
    return str(abs(hash(frozenset(my_dict.items()))))

def run_job(job):
    script, kwargs = job
    target = path.join(kwargs['output_dir'], kwargs['name'])+ ".json"

    if not path.exists(target):

        # Escape \ with \\ for bash escaping
        for key, value in iteritems(kwargs):
            if isinstance(value, str):
                kwargs[key] = value.replace(r'"', r'\"')

        cmd = "{} {}".format(script, " ".join("--{}={}".format(k, v) for k, v in iteritems(kwargs)))
        logger.info("Running " + cmd)
        res = system(cmd)
        if res != 0:
            logger.error("Failed job {}".format(cmd))
            with open(path.join(kwargs['output_dir'], "failed_jobs.err"), "a") as f:
                f.write("{}\n".format(cmd))
    else:
        done_job = json.load(open(target))
        shared_items = set(kwargs.items()) & set(done_job['opts'].items())
        if not (len(shared_items) == len(kwargs) == len(done_job['opts'])):
            logger.error("Wrote down job has non matching json")
            with open(path.join(kwargs['output_dir'], "duplicated_jobs.err"), "a") as f:
                f.write("{}\n".format(target + ".json"))

def wac_score(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)
    if cm.shape != (2,2):
        return accuracy_score(Y_true, Y_pred)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    if tp == 0 and fn == 0:
        return 0.5*tn/float(tn+fp)
    elif tn == 0 and fp == 0:
        return 0.5*tp/float(tp+fn)
    return 0.5*tp/float(tp+fn) + 0.5*tn/float(tn+fp)

def wac_scoring(estimator, X, y):
    return wac_score(y, estimator.predict(X))


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


def get_mean_experiments_results(results_dir, batch_sizes=[20, 50, 100], strategies='all'):
    """
    Read all experiments from given directory, calculated mean results for all combinations of
    strategies and batch_sizes
    :param results_dir: string, path to directiory with results
    :param batch_sizes: list of ints, batch sizes the experiments were run on
    :param strategies: string, strategies the experimtens were run on, default 'all'
    :return: dict of dicts: {strategy-batchsize: {metric: mean_results}}
    """

    if strategies == 'all':
        strategies = ['UncertaintySampling', 'PassiveStrategy', 'QuasiGreedyBatch', 'QueryByBagging']

    mean_scores = {strategy + '-' + str(batch_size): defaultdict(list) for strategy in strategies for batch_size in batch_sizes}

    for results_file in filter(lambda x: x[-4:] == 'json', os.listdir(results_dir)):

        json_path = os.path.join(results_dir, results_file)
        assert os.path.exists(json_path)

        json_results = load_json(json_path)
        strategy = json_results['opts']['strategy']
        batch_size = json_results['opts']['batch_size']

        key = strategy + "-" + str(batch_size)
        assert key in mean_scores

        pkl_path = os.path.join(results_dir, results_file[:-5] + ".pkl.gz")
        assert os.path.exists(pkl_path)

        with gzip.open(os.path.join(results_dir, pkl_path), 'r') as f:
            scores = cPickle.load(f)

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
                mean_scores[strategy][metric] = np.vstack(values).mean(axis=0)
            except Exception:
                import pdb
                pdb.set_trace()
            assert mean_scores[strategy][metric].shape[0] > 1

    return mean_scores


def compare_curves(results_dir, scores, metrics=['wac_score_valid'], batch_sizes=[20, 50, 100]):
    """
    Plot curves from given mean scores and metrics
    :param results_dir: string, path to directiory with results
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

    # plot all passed metrics
    for ax, (batch_size, metric) in zip(axes, exp_type):
        for strategy, score in scores.iteritems():
            if strategy.split('-')[1] == str(batch_size):
                strategy_name = strategy.split('-')[0]
                pd.DataFrame({strategy_name: score[metric]}).plot(title='%s %d batch size' % (metric, batch_size), ax=ax)
                ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5))


def plot_curves(results_dir, metrics):
    """
    Plots curves for mean results off all experiments in given directory for given metrics
    :param results_dir: string, path to results directory
    :param metrics: list of strings or string, which metrics to plot
    :return:
    """
    mean_scores = get_mean_experiments_results(results_dir)
    compare_curves(results_dir, mean_scores, metrics=metrics)


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