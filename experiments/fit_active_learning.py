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
import traceback
import sys
from sacred import Experiment
from misc.config import *
from kaggle_ninja import *
from utils import ExperimentResults, binary_metrics
from experiment_runner import fit_AL_on_folds
from collections import defaultdict
from itertools import chain
from functools import partial
ex = Experiment("fit_active_learning")
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

@ex.config
def my_config():
    experiment_detailed_name = "active_uncertanity_sampling"
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
    base_model = "SGDClassifier"
    base_model_kwargs = {}
    strategy= "random_query"
    strategy_kwargs={}

@ex.capture
def run(experiment_detailed_name, strategy_kwargs, batch_size, fingerprint, strategy, protein,\
        base_model, base_model_kwargs, \
        preprocess_fncs, loader_function, loader_args, seed, _log, _config):
    strategy_kwargs = copy.deepcopy(strategy_kwargs)
    loader_args = copy.deepcopy(loader_args)
    loader_function = copy.deepcopy(loader_function)
    base_model_kwargs = copy.deepcopy(base_model_kwargs)

    ## Prepare data loader ##
    loader = [loader_function, loader_args]
    comp = [[protein, fingerprint]]
    print _config
    print base_model

    if base_model not in globals():
        raise ValueError("Not imported base_model class into global namespace. Aborting")

    base_model_cls = partial(globals()[base_model], random_state=seed, **base_model_kwargs)
    strategy = partial(find_obj(strategy), **strategy_kwargs)
    model_cls = partial(ActiveLearningExperiment, strategy=strategy, base_model_cls=base_model_cls, batch_size=batch_size)

    folds, _, _ = get_data(comp, loader, preprocess_fncs).values()[0]

    metrics, monitors = fit_AL_on_folds(model_cls, folds)

    return ExperimentResults(results=dict(metrics), monitors=monitors, dumps={}, \
                             config=_config, name=experiment_detailed_name)


## Needed boilerplate ##

@ex.main
def main(experiment_detailed_name, timeout, loader_args, seed, force_reload, _log):
    try:
        ex.logger = get_logger(experiment_detailed_name)

        loader_args['seed'] = seed # This is very important to keep immutable config afterwards
        _log.info("Fitting  "+experiment_detailed_name + " force_reload="+str(force_reload))


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
    except Exception, err:
        _log.error(traceback.format_exc())
        _log.error(sys.exc_info()[0])
        raise(err)

@ex.capture
def save(results, experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    print "Saving ", _config
    ninja_set_value(value=results, master_key=experiment_detailed_name, **_config_cleaned)

@ex.capture
def try_load(experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    print "Loading ", _config
    return ninja_get_value(master_key=experiment_detailed_name, **_config_cleaned)

if __name__ == '__main__':
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("fit_active_learning", ex)
