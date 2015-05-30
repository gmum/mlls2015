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
import experiments
from experiments import utils
from kaggle_ninja import *
from experiments.utils import ExperimentResults
from experiments.experiment_runner import run_experiment

import random_query

ex = Experiment('random_query_composite')

@ex.config
def my_config():
    experiment_sub_name = "random_query_composite"
    base_batch_size = 10
    seed = 778
    timeout = -1
    force_reload = False

@ex.capture
def run(experiment_sub_name, base_batch_size, seed, _log, _config):
    val1 = run_experiment("random_query_exp", batch_size=base_batch_size, seed=seed)
    val2 = run_experiment("random_query_exp", batch_size=2*base_batch_size, seed=seed)
    return ExperimentResults(name=ex.name,\
                             monitors={}, results={"acc": val1.results["acc"] + val2.results["acc"]}, dumps={}, config=_config)

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
def save(results, experiment_sub_name, _config, _log):
    _log.info(results)
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    ninja_set_value(value=results, master_key=experiment_sub_name, **_config_cleaned)

@ex.capture
def try_load(experiment_sub_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    return ninja_get_value(master_key=experiment_sub_name, **_config_cleaned)

if __name__ == '__main__':
    ex.logger = get_logger("al_ecml")
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("random_query_composite", ex)