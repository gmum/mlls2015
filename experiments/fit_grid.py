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
    ipcluster_workers = 0
    force_reload=False
    n_jobs = 4
    recalculate_experiments = False
    timeout = -1
    single_fit_timeout = -1
    seed = 777

@ex.capture
def run(recalculate_experiments, experiment_detailed_name, seed, n_jobs, single_fit_timeout, ipcluster_workers, \
        _config, grid_params, base_experiment, base_experiment_kwargs, _log):
    logger = get_logger(experiment_detailed_name)
    logger.info("Fitting grid for "+base_experiment + " recalcualte_experiments="+str(recalculate_experiments))

    if ipcluster_workers == 0:
        ipcluster_workers = None
    else:
        from IPython.parallel import Client
        c = Client()
        ipcluster_workers = [c[id] for id in ipcluster_workers]

    start_time = time.time()
    experiments = run_experiment_grid(base_experiment, logger=logger, ipcluster_workers=ipcluster_workers,
                                      force_reload=recalculate_experiments, seed=seed, timeout=single_fit_timeout, \
                                      experiment_detailed_name=experiment_detailed_name, \
                                      n_jobs=n_jobs, grid_params=grid_params, **base_experiment_kwargs)
    misc = {'grid_time': time.time() - start_time}

    return GridExperimentResult(experiments=experiments, misc=misc,
                                config=_config, grid_params=grid_params, name=experiment_detailed_name)


## Needed boilerplate ##


@ex.main
def main(experiment_detailed_name, base_experiment_kwargs, recalculate_experiments, timeout, force_reload):
    try:
        logger = get_logger(experiment_detailed_name)

        force_reload = recalculate_experiments or force_reload
        assert('seed' not in base_experiment_kwargs) # We don't want to repeat having seed in many places. This is confusing

        # Load cache unless forced not to
        cached_result = try_load() if not force_reload else None
        if cached_result:
            logger.info("Read from cache "+ex.name)
            return cached_result
        else:
            logger.info("Cache miss, calculating")
            if timeout > 0:
                result = abortable_worker(run, timeout=timeout)
            else:
                result = run()
            save(result)
            return result
    except Exception, err:
        logger.error(traceback.format_exc())
        logger.error(sys.exc_info()[0])
        raise(err)
@ex.capture
def save(results, experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    del _config_cleaned['n_jobs']
    del _config_cleaned['ipcluster_workers']
    del _config_cleaned['recalculate_experiments']
    ninja_set_value(value=results, master_key=experiment_detailed_name, **_config_cleaned)

@ex.capture
def try_load(experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    del _config_cleaned['ipcluster_workers']
    del _config_cleaned['n_jobs']
    del _config_cleaned['recalculate_experiments']
    return ninja_get_value(master_key=experiment_detailed_name, **_config_cleaned)

if __name__ == '__main__':
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("fit_grid", ex)
