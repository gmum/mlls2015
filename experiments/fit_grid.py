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
import traceback, sys
import fit_active_learning

ex = Experiment("fit_grid")
from sklearn.linear_model import SGDClassifier


@ex.config
def my_config():
    experiment_detailed_name = "uncertanity_sampling"
    base_experiment = "fit_active_learning"
    base_experiment_kwargs = {}
    grid_params = {}
    force_reload=False
    n_jobs = 4
    recalculate_experiments = False
    timeout = -1
    single_fit_timeout = -1
    seed = 777

@ex.capture
def run(recalculate_experiments, experiment_detailed_name, seed, n_jobs, single_fit_timeout, \
        _config, grid_params, base_experiment, base_experiment_kwargs, _log):
    ex.logger.info("Fitting grid for "+base_experiment + " recalcualte_experiments="+str(recalculate_experiments))


    experiments = run_experiment_grid(base_experiment, logger=ex.logger,
                                      force_reload=recalculate_experiments, seed=seed, timeout=single_fit_timeout, \
                                      experiment_detailed_name=experiment_detailed_name, \
                                      n_jobs=n_jobs, grid_params=grid_params, **base_experiment_kwargs)

    return GridExperimentResult(experiments=experiments, config=_config, grid_params=grid_params, name=experiment_detailed_name)


## Needed boilerplate ##

@ex.main
def main(base_experiment_kwargs, recalculate_experiments, experiment_detailed_name, timeout, seed, force_reload, _log):
    try:
        ex.logger = get_logger(experiment_detailed_name)

        force_reload = recalculate_experiments or force_reload
        assert('seed' not in base_experiment_kwargs) # We don't want to repeat having seed in many places. This is confusing

        # Load cache unless forced not to
        cached_result = try_load() if not force_reload else None
        if cached_result:
            ex.logger.info("Read from cache "+ex.name)
            return cached_result
        else:
            ex.logger.info("Cache miss, calculating")
            if timeout > 0:
                result = abortable_worker(run, timeout=timeout)
            else:
                result = run()
            save(result)
            return result
    except Exception, err:
        ex.logger.error(traceback.format_exc())
        ex.logger.error(sys.exc_info()[0])
        raise(err)
@ex.capture
def save(results, experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    del _config_cleaned['n_jobs']
    del _config_cleaned['recalculate_experiments']
    print "Saving ", _config_cleaned
    ninja_set_value(value=results, master_key=experiment_detailed_name, **_config_cleaned)

@ex.capture
def try_load(experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    del _config_cleaned['n_jobs']
    del _config_cleaned['recalculate_experiments']
    print "Loading ", _config_cleaned
    return ninja_get_value(master_key=experiment_detailed_name, **_config_cleaned)

if __name__ == '__main__':
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("fit_grid", ex)
