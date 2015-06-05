import sys
sys.path.append("..")
import misc
from misc.config import *
from kaggle_ninja import *
from collections import namedtuple
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, auc, \
    mean_absolute_error, confusion_matrix, precision_score, recall_score, matthews_corrcoef
import pandas as pd



def jaccard_similarity_score_fast(r1, r2):
    dt = float(r1.dot(r2.T).sum())
    return dt / (r1.sum() + r2.sum() - dt )

def g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2,2):
        return accuracy_score(y_true, y_pred)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    return np.sqrt((tp / float(tp+fn)) * (tn / float(tn+fp)))

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

ExperimentResults = namedtuple("ExperimentResults", ["results", "dumps", "monitors", "name", "config"])
GridExperimentResult = namedtuple("GridExperimentResult", ["experiments", "grid_params", "name", "config"])

import matplotlib.pylab as plt

def get_best(experiments, metric):
    return sorted(experiments, key=lambda x: x.results.get(metric, 0))[-1]

def calc_auc(experiments, exclude=['iter', 'n_already_labeled'], folds='all'):
    assert folds in ['all', 'mean']

    if not isinstance(experiments, list):
        experiments = [experiments]
    for e in experiments:
        if not isinstance(e.monitors, list):
            e.monitors = [e.monitors]

    assert(all(len(e.monitors) == len(experiments[0].monitors) for e in experiments))
    keys = [k for k in experiments[0].monitors[0].keys() if k not in exclude]
    n_iter = experiments[0].monitors[0]['iter']
    n_folds = len(experiments[0].monitors)

    if len(experiments[0].monitors[0]) > 1 and folds == 'mean':
        for i, e in enumerate(experiments):
            mean_monitors = {k: np.zeros(n_iter) for k in keys}
            for fold_mon in e.monitors:
                for k in keys:
                    if len(fold_mon[k]) + 1 == n_iter:
                        fold_mon[k].append(fold_mon[k][-1])
                    assert len(fold_mon[k]) == n_iter, "monitor for %s is length %i while n_iter is %i" % (k, len(fold_mon[k]), n_iter)
                    mean_monitors[k] += np.array(fold_mon[k])
            for k, v in mean_monitors.iteritems():
                mean_monitors[k] = v / n_folds
            experiments[i] = e._replace(monitors=[mean_monitors])

    for key in keys:
        for e in experiments:
            for i, mon in enumerate(e.monitors):
                area = auc(np.arange(n_iter), mon[key])
                print "%s %s: %f" % (e.name + str(i), key, area)

        print

def plot_monitors(experiments, exclude=['iter', 'n_already_labeled'], folds='all'):

    assert folds in ['all', 'mean']

    if not isinstance(experiments, list):
        experiments = [experiments]
    for e in experiments:
        if not isinstance(e.monitors, list):
            e.monitors = [e.monitors]

    assert(all(len(e.monitors) == len(experiments[0].monitors) for e in experiments))
    keys = [k for k in experiments[0].monitors[0].keys() if k not in exclude]
    n_iter = experiments[0].monitors[0]['iter']
    n_folds = len(experiments[0].monitors)

    if len(experiments[0].monitors[0]) > 1 and folds == 'mean':
        for i, e in enumerate(experiments):
            mean_monitors = {k: np.zeros(n_iter) for k in keys}
            for fold_mon in e.monitors:
                for k in keys:
                    if len(fold_mon[k]) + 1 == n_iter:
                        fold_mon[k].append(fold_mon[k][-1])
                    assert len(fold_mon[k]) == n_iter, "monitor for %s is length %i while n_iter is %i" % (k, len(fold_mon[k]), n_iter)
                    mean_monitors[k] += np.array(fold_mon[k])
            for k, v in mean_monitors.iteritems():
                mean_monitors[k] = v / n_folds
            experiments[i] = e._replace(monitors=[mean_monitors])

    f, axes = plt.subplots(len(keys), 1)
    f.set_figheight(15)
    f.set_figwidth(15)
    for ax, key in zip(axes, keys):
        for e in experiments:
            for i, mon in enumerate(e.monitors):
                pd.DataFrame({e.name + str(i): mon[key]}).plot(title=key, ax=ax)


def plot_grid_experiment_results(grid_results, params, metrics):
    global plt
    params = sorted(params)
    grid_params = grid_results.grid_params
    plt.figure(figsize=(8, 6))
    for metric in metrics:
        grid_params_shape = [len(grid_params[k]) for k in sorted(grid_params.keys())]
        params_max_out = [(1 if k in params else 0) for k in sorted(grid_params.keys())]
        results = np.array([e.results.get(metric, 0) for e in grid_results.experiments])
        results = results.reshape(*grid_params_shape)
        for axis, included_in_params in enumerate(params_max_out):
            if not included_in_params:
                results = np.apply_along_axis(np.max, axis, results)

        print results
        params_shape = [len(grid_params[k]) for k in sorted(params)]
        results = results.reshape(*params_shape)

        if len(results.shape) == 1:
            results = results.reshape(-1,1)
        import matplotlib.pylab as plt

        #f.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(results, interpolation='nearest', cmap=plt.cm.hot)
        plt.title(str(grid_results.name) + " " + metric)

        if len(params) == 2:
            plt.xticks(np.arange(len(grid_params[params[1]])), grid_params[params[1]], rotation=45)
        plt.yticks(np.arange(len(grid_params[params[0]])), grid_params[params[0]])
        plt.colorbar()
        plt.show()


def print_experiment_results(experiments, metrics=["mean_mcc_valid"]):
    """
    @param experiments list of ExperimentResults
    @param metrics metrics to pick from ExperimentResults.results
    """
    rows = []
    for e in experiments: #sorted(l, key=lambda x: x['arguments']['args']['protein']):
        rows.append([e.results.get(m, None) for m in metrics])
    return pd.DataFrame(rows, columns=metrics, index=[e.name for e in experiments])

def binary_metrics(Y_true, Y_pred, dataset_name):
    assert(Y_true.shape == Y_pred.shape)
    metrics = {"wac": wac_score(Y_true, Y_pred), "mcc": matthews_corrcoef(Y_true, Y_pred),\
            "precision": precision_score(Y_true,Y_pred), "recall": recall_score(Y_true, Y_pred)
            }
    return {k+"_"+dataset_name : v for k,v in metrics.items() }
