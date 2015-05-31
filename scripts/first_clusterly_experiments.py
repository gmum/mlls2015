import sys
sys.path.append("..")
import pandas as pd
from misc.config import *
import logging
from models.strategy import *
from experiments import experiment_runner, fit_active_learning, fit_grid
from kaggle_ninja import *
from experiments.utils import *
from experiment_runner import run_experiment
from get_data import *
from get_data import _get_raw_data, fingerprints, proteins
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from get_data import *

preprocess_fncs = [["to_binary", {"all_below": True}]]
loader = ["get_splitted_data_clusterly", {
        "seed": 777, "preprocess_fncs": preprocess_fncs, "n_folds": 2}]
data = get_data([["5ht6", "KlekFP"]], loader, preprocess_fncs, force_reload=True)

grid_results = run_experiment("fit_grid",
                       n_jobs = 2, \
                       experiment_detailed_name="5ht6_KlekFP_uncertanity",
                       base_experiment="fit_active_learning",
                       seed=777,
                       protein="5ht6",
                       fingerprint="KlekFP",
                       grid_params = {"base_model_kwargs:alpha": list(np.logspace(-5,5,10))},
                       base_experiment_kwargs={"strategy": "random_query",
                                               "loader_function": loader[0],
                                               "loader_args": loader[1],
                                               "batch_size": 20, \
                                               "base_model": "SGDClassifier"})

