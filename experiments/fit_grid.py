"""
Example usage: python fit_grid.py with n_jobs=2 grid_params="{'alpha': [1e-3, 1e-2, 10, 100]}"
"""

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
from utils import ExperimentResults, GridExperimentResult, binary_metrics
from experiment_runner import fit_AL_on_folds, run_experiment_grid
from collections import defaultdict
from itertools import chain

import fit_active_learning
import fit_svm

ex = Experiment("fit_grid")
from sklearn.linear_model import SGDClassifier


@ex.config
def my_config():
    experiment_detailed_name = "uncertanity_sampling"
    base_experiment = "fit_active_learning"
    base_experiment_kwargs = {}
    grid_params = {}
    force_reload=False
    loader_function = "get_splitted_data"
    loader_args = {"n_folds": 2,
               "test_size":0.0}
    n_jobs = 4
    timeout = -1
    single_fit_timeout = -1
    seed = 777

@ex.capture
def run(experiment_detailed_name, seed, n_jobs, single_fit_timeout, _config, grid_params, base_experiment, base_experiment_kwargs, _log):
    _log.info("Fitting grid for "+base_experiment)


    experiments = run_experiment_grid(base_experiment, seed=seed, timeout=single_fit_timeout, \
                                      experiment_detailed_name=experiment_detailed_name+"_subfit", \
                                      n_jobs=n_jobs, grid_params=grid_params, **base_experiment_kwargs)

    return GridExperimentResult(experiments=experiments, config=_config, grid_params=grid_params, name=experiment_detailed_name)


## Needed boilerplate ##

@ex.main
def main(experiment_detailed_name, timeout, loader_args, seed, force_reload, _log):
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
def save(results, experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    del _config_cleaned['n_jobs']
    print "Saving ", _config
    ninja_set_value(value=results, master_key=experiment_detailed_name, **_config_cleaned)

@ex.capture
def try_load(experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    del _config_cleaned['n_jobs']
    print "Loading ", _config
    return ninja_get_value(master_key=experiment_detailed_name, **_config_cleaned)

if __name__ == '__main__':
    ex.logger = main_logger
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("fit_grid", ex)
