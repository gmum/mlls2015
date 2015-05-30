import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from get_data import get_data
from models.active_model import ActiveLearningExperiment
from models.strategy import random_query
from models.utils import ObstructedY
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import copy
from sacred import Experiment
from misc.config import *
from kaggle_ninja import *
import experiments
from experiments import utils
from experiments.utils import ExperimentResults, binary_metrics
from experiments.experiment_runner import fit_AL_on_folds
from collections import defaultdict
from itertools import chain
ex = Experiment('random_query')

@ex.config
def my_config():
    experiment_name = "random_query"
    batch_size = 10
    seed = 778
    timeout = -1

    ## Not defining experiment ##
    force_reload = False


    ## Dataset ##
    fingerprint = 'ExtFP'
    protein = '5ht7'
    loader_function = "get_splitted_data"
    loader_args = {"n_folds": 10,
               "seed":-1,
               "test_size":0.0}
    preprocess_fncs = [["to_binary", {"all_below": True}]]

@ex.capture
def run(batch_size, fingerprint, protein, preprocess_fncs, loader_function, loader_args, seed, _log):
    time.sleep(2) # Please don't remove, important for tests ..
    loader = [loader_function, loader_args]
    comp = [[protein, fingerprint]]
    loader[1]['seed'] = seed

    sgd = SGDClassifier(random_state=seed)
    strat = partial(random_query, model=None)
    model = ActiveLearningExperiment(strategy=strat, base_model=sgd, batch_size=batch_size)

    folds, _, _ = get_data(comp, loader, preprocess_fncs).values()[0]

    metrics = fit_AL_on_folds(model, folds)

    print metrics

    return ExperimentResults(results=metrics, monitors={}, dumps={})


## Needed boilerplate ##

@ex.main
def main(timeout, force_reload, _log):
    # Load cache unless forced not to
    cached_result = try_load() if not force_reload else None
    if cached_result:
        _log.info("Reading from cache "+ex.name)
        return cached_result
    else:
        if timeout > 0:
            result = abortable_worker(run, timeout=timeout)
        else:
            result = run()
        save(result)
        return result

@ex.capture
def save(results, experiment_name, _config, _log):
    _log.info(results)
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    ninja_set_value(value=results, master_key=experiment_name, **_config_cleaned)

@ex.capture
def try_load(experiment_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    return ninja_get_value(master_key=experiment_name, **_config_cleaned)

if __name__ == '__main__':
    ex.logger = get_logger("al_ecml")
    results = ex.run_commandline().result
    save(results)

import kaggle_ninja
kaggle_ninja.register("random_query_exp", ex)
