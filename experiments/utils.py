import sys
sys.path.append("..")
import misc
from misc.config import *
from kaggle_ninja import *
from collections import namedtuple
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, \
    mean_absolute_error, confusion_matrix, precision_score, recall_score, matthews_corrcoef
import pandas as pd

def jaccard_similarity_score_fast(r1, r2):
    dt = float(r1.dot(r2.T).sum())
    return dt / (r1.sum() + r2.sum() - dt )



def wac_score(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)
    assert(cm.shape==(2,2))
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    return 0.5*tp/float(tp+fn) + 0.5*tn/float(tn+fp)

ExperimentResults = namedtuple("ExperimentResults", ["results", "dumps", "monitors", "name", "sub_name"])
GridExperimentResult = namedtuple("GridExperimentResult", ["experiments", "grid_params", "name", "sub_name"])


def print_experiment_results(experiments, metrics=["mean_mcc_valid"]):
    """
    @param experiments list of ExperimentResults
    @param metrics metrics to pick from ExperimentResults.results
    """
    rows = []
    for e in experiments: #sorted(l, key=lambda x: x['arguments']['args']['protein']):
        rows.append([e.results.get(m, None) for m in metrics])
    return pd.DataFrame(rows, columns=metrics, index=[e.name + "_" + e.sub_name for e in es])

def binary_metrics(Y_true, Y_pred, dataset_name):
    assert(Y_true.shape == Y_pred.shape)
    metrics = {"wac": wac_score(Y_true, Y_pred), "mcc": matthews_corrcoef(Y_true, Y_pred),\
            "precision": precision_score(Y_true,Y_pred), "recall": recall_score(Y_true, Y_pred)
            }
    return {k+"_"+dataset_name : v for k,v in metrics.items() }
