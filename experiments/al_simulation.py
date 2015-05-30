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
from utils import ExperimentResults, binary_metrics
from experiment_runner import fit_AL_on_folds
from collections import defaultdict
from itertools import chain
ex = Experiment('al_simulation')

@ex.config
def my_config():
    experiment_name = "al_simulation"
    batch_size = 10
    seed = 778
    timeout = -1
    force_reload = False
    fingerprint = 'ExtFP'
    protein = '5ht7'
    loader_function = "get_splitted_data"
    loader_args = {"n_folds": 2,
               "test_size":0.0}
    preprocess_fncs = []
    strategy= "uncertanity_sampling"

@ex.capture
def run(batch_size, fingerprint, strategy, protein, preprocess_fncs, loader_function, loader_args, seed, _log):
    loader_args = copy.deepcopy(loader_args)
    loader_function = copy.deepcopy(loader_function)


    ## Prepare data loader ##
    loader = [loader_function, loader_args]
    comp = [[protein, fingerprint]]


    sgd = SGDClassifier(random_state=seed)
    model = ActiveLearningExperiment(strategy=find_obj(strategy), base_model=sgd, batch_size=batch_size)

    folds, _, _ = get_data(comp, loader, preprocess_fncs).values()[0]

    metrics = fit_AL_on_folds(model, folds)

    return ExperimentResults(results=dict(metrics), monitors={}, dumps={})


## Needed boilerplate ##

@ex.main
def main(timeout, loader_args, seed, force_reload, _log):
    loader_args['seed'] = seed # This is very important to keep immutable config afterwards

    # Load cache unless forced not to
    cached_result = try_load() if not force_reload else None
    if cached_result:
        _log.info("Read from cache "+ex.name)
        return cached_result
    else:
        _log.info("Cache miss, calculating")
        if timeout > 0:
            result = abortable_worker(run, timeout=timeout)
        else:
            result = run()
        save(result)
        return result

@ex.capture
def save(results, experiment_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    print "Saving ", _config
    ninja_set_value(value=results, master_key=experiment_name, **_config_cleaned)

@ex.capture
def try_load(experiment_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    print "Loading ", _config
    return ninja_get_value(master_key=experiment_name, **_config_cleaned)

if __name__ == '__main__':
    ex.logger = get_logger("al_ecml")
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("al_simulation", ex)
