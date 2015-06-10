#TODO: integration test for fixed projection

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
from sacred import Experiment
from misc.config import *
from kaggle_ninja import *
from utils import ExperimentResults, binary_metrics
from experiment_runner import fit_AL_on_folds
ex = Experiment("fit_active_learning")
from sklearn.metrics import auc
from sklearn.linear_model import SGDClassifier
from models.balanced_models import *

@ex.config
def my_config():
    experiment_detailed_name = "active_uncertanity_sampling"
    batch_size = 10
    seed = -1
    timeout = -1
    id_folds = -1
    warm_start_percentage = 0
    force_reload = False

    # Required args. could comment out
    # but this way raising more visible errors
    fingerprint = 0
    protein = 0
    loader_function = 0
    loader_args = 0
    preprocess_fncs = 0

    base_model = "SGDClassifier"
    base_model_kwargs = {}
    strategy= "random_query"
    strategy_projection_h = 0
    strategy_kwargs={}
    param_grid={}

@ex.capture
def run(experiment_detailed_name, warm_start_percentage, strategy_kwargs, id_folds, strategy_projection_h,
        batch_size, fingerprint, strategy, protein,\
        base_model, base_model_kwargs, param_grid, \
        preprocess_fncs, loader_function, loader_args, seed, _config):

    logger = get_logger(experiment_detailed_name)

    assert preprocess_fncs != 0, "Please pass preprocess_fncs"
    assert loader_function != 0, "Please pass loader_function"
    assert loader_args != 0, "Please pass loader_args"
    assert protein != 0, "Please pass protein"
    assert fingerprint != 0, "Please pass fingerprint"
    assert seed != -1, "Please pass seed"

    strategy_kwargs = copy.deepcopy(strategy_kwargs)
    loader_args = copy.deepcopy(loader_args)
    loader_function = copy.deepcopy(loader_function)
    base_model_kwargs = copy.deepcopy(base_model_kwargs)

    ## Prepare data loader ##
    loader = [loader_function, loader_args]
    comp = [[protein, fingerprint]]

    if base_model not in globals():
        raise ValueError("Not imported base_model class into global namespace. Aborting")

    # Construct model with fixed projection
    base_model_cls = globals()[base_model]

    if "h" in param_grid:
        projector_cls = partial(FixedProjector, h_max=max(param_grid["h"]), projector=RandomProjector())
    else:
        projector_cls = None

    strategy = find_obj(strategy)

    model_cls = partial(ActiveLearningExperiment, logger=logger,
                        strategy=strategy, batch_size=batch_size,
                        strategy_kwargs=strategy_kwargs, param_grid=param_grid)

    folds, _, _ = get_data(comp, loader, preprocess_fncs).values()[0]

    logger.info("Fitting on loader "+str(loader) + " preprocess_fncs="+str(preprocess_fncs))
    logger.info(folds[0]["X_train"]["data"].shape)

    metrics, monitors = fit_AL_on_folds(model_cls=model_cls, base_model_cls=base_model_cls, base_model_kwargs=base_model_kwargs, \
                                        projector_cls=projector_cls,\
                                        folds=folds, logger=logger, id_folds=id_folds,
                                        base_seed=seed, warm_start_percentage=warm_start_percentage)
    misc = {}
    if id_folds == -1 or len(id_folds) == len(folds):
        mean_monitor = {k: np.zeros(len(v)) for k, v in monitors[0].iteritems() if isinstance(v, list)}

        for fold_monitor in monitors:
            for key in mean_monitor.keys():
                mean_monitor[key] += np.array(fold_monitor[key])

        for key, values in dict(mean_monitor).iteritems():
            mean_monitor[key] = values / len(monitors)
            metrics['auc_' + key] = auc(np.arange(values.shape[0]), values)

        misc = {'mean_monitor': mean_monitor}


    misc['X_train_size'] = folds[0]["X_train"]["data"].shape
    misc['X_valid_size'] = folds[0]["X_valid"]["data"].shape

    logger.info("Logging following keys in monitors: "+str(monitors[0].keys()))

    return ExperimentResults(results=dict(metrics), misc=misc, monitors=monitors, dumps={}, \
                             config=_config, name=experiment_detailed_name)


## Needed boilerplate ##

@ex.main
def main(experiment_detailed_name, timeout, loader_args, seed, force_reload):
    try:
        logger = get_logger(experiment_detailed_name)

        loader_args['seed'] = seed # This is very important to keep immutable config afterwards
        logger.info("Fitting  "+experiment_detailed_name + " force_reload="+str(force_reload))

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

    ninja_set_value(value=results, master_key=experiment_detailed_name, **_config_cleaned)

@ex.capture
def try_load(experiment_detailed_name, _config, _log):
    _config_cleaned = copy.deepcopy(_config)
    del _config_cleaned['force_reload']
    return ninja_get_value(master_key=experiment_detailed_name, **_config_cleaned)

if __name__ == '__main__':
    results = ex.run_commandline().result

import kaggle_ninja
kaggle_ninja.register("fit_active_learning", ex)
