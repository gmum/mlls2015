import sys
sys.path.append("..")
import misc
from misc.config import *
from kaggle_ninja import *
from collections import namedtuple
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, auc, \
    mean_absolute_error, confusion_matrix, precision_score, recall_score, matthews_corrcoef
import pandas as pd
from collections import defaultdict
import copy

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

ExperimentResults = namedtuple("ExperimentResults", ["results", "dumps" ,"misc",  "monitors", "name", "config"])
GridExperimentResult = namedtuple("GridExperimentResult", ["experiments", "misc",  "grid_params", "name", "config"])

def get_best(experiments, metric):
    return sorted(experiments, key=lambda x: x.results[metric])[-1]

def get_experiment_results(experiment_detailed_name):
    # This is very hacky, and very well needed. We will refactor it into database query
    f_name = \
        sorted(glob.glob(os.path.join(c["BASE_DIR"], "*"+experiment_detailed_name+"*.pkl")), key=lambda k: len(k))[0]
    return pickle.load(open(f_name, "r"))

def dashboard(finished=False):
    """
    Prints out running experiments on given machine
    """

    import glob
    import json
    jsons = []

    for f in glob.glob(os.path.join(c["BASE_DIR"], "*.info")):
        with open(f, "r") as fh:
            try:
                js = json.loads(fh.read())
                if not finished:
                    if time.time() - js.get('heartbeat',0) < 60: # Last hearbeat within 60s
                        jsons.append(js)
                else:
                    if js.get("progress", 0.0) == 1.0:
                        jsons.append(js)

                if "call_time" not in jsons[-1]:
                    jsons[-1]["call_time"] = ""

            except:
                pass

    return pd.DataFrame(jsons)

def plot_monitors(experiments, keys='metrics', folds='mean', figsize=(15,15)):
    import matplotlib.pylab as plt

    # Hack for igor
    experiments = copy.deepcopy(experiments)

    assert folds in ['all', 'mean']
    assert keys in ['metrics', 'times']

    if keys == 'times':
        include = ['unlabeled_test_times',
                   'grid_times',
                   'strat_times',
                   'concept_test_times']
    elif keys == "metrics":
        include = []
        for metr in ["wac_score", "matthews_corrcoef"]:
            for dataset in ["concept", "unlabeled", "cluster_A_concept", "cluster_B_concept",
                            "cluster_A_unlabeled", "cluster_B_unlabeled"]:
                include.append(metr+"_"+dataset)


    # TODO: ??
    if not isinstance(experiments, list):
        experiments = [experiments]
    for e in experiments:
        if not isinstance(e.monitors, list):
            e.monitors = [e.monitors]

    assert(all(len(e.monitors) == len(experiments[0].monitors) for e in experiments))
    keys = [k for k in experiments[0].monitors[0].keys() if k in include]

    f, axes = plt.subplots(len(keys), 1)
    f.set_figheight(figsize[1])
    f.set_figwidth(figsize[0])

    if folds == 'mean':
        for ax, key in zip(axes, keys):
            for e in experiments:
                pd.DataFrame({e.name: e.misc['mean_monitor'][key]}).plot(title=key, ax=ax)
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    else:
        for ax, key in zip(axes, keys):
            for e in experiments:
                for mon in e.monitors:
                    pd.DataFrame({e.name: mon[key]}).plot(title=key, ax=ax)
                    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))




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
