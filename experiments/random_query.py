import sys
sys.path.append("..")

from get_data import get_data
from models.active_model import ActiveModel
from models.strategy import random_query
from models.utils import ObstructedY

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from sacred import Experiment
from misc.config import *
from kaggle_ninja import *
from utils import ExperimentResults

ex = Experiment('random_query')

@ex.config
def my_config():
    batch_size = 10
    seed = 777

@ex.capture
def run(batch_size, seed, _log):
    seed = 666
    strategy_args = {'batch_size': batch_size, 'seed': seed}
    comp = [['5ht7', 'ExtFP']]
    loader = ["get_splitted_data",
              {"n_folds": 3,
               "seed":seed,
               "test_size":0.1}]
    preprocess_fncs = []

    sgd = SGDClassifier(random_state=seed)
    model = ActiveModel(strategy=random_query, base_model=sgd)

    folds, test_data, data_desc = get_data(comp, loader, preprocess_fncs).values()[0]
    _log.info(data_desc)

    X = folds[0]['X_train']
    y = ObstructedY(folds[0]['Y_train'])

    X_test = folds[0]['X_valid']
    y_test = folds[0]['Y_valid']

    model.fit(X, y, strategy_args=strategy_args, verbose=True)
    p = model.predict(X_test)
    return ExperimentResults(results={"acc": accuracy_score(p, y_test)}, monitors={}, dumps={})


@ex.main
def main(_log):
    cached_result = try_load()
    if cached_result:
        _log.info("Reading from cache "+ex.name)
        return cached_result
    else:
        return run()

@ex.capture
def save(results, _config, _log):
    _log.info(results)
    ninja_set_value(value=results, name=ex.name, **_config)

@ex.capture
def try_load(_config, _log):
    return ninja_get_value(name=ex.name, **_config)

if __name__ == '__main__':
    ex.logger = get_logger("al_ecml")
    results = ex.run_commandline().result
    save(results)