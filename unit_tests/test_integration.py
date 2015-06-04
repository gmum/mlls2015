# Whole architecture test on [-1,-1]^2 dataset

import sys
sys.path.append("..")
import kaggle_ninja
from kaggle_ninja import *
import random_query, random_query_composite
from experiments import experiment_runner, fit_active_learning, fit_grid
from experiments.experiment_runner import run_experiment, run_experiment_grid
from experiments.utils import plot_grid_experiment_results, get_best, plot_monitors
from misc.config import *
# from experiment_runner import _replace_in_json
import seaborn

import unittest
import os

turn_on_force_reload_all()

class TestDataAPI(unittest.TestCase):


    def test_checkerboard(self):
        grid_results_uncert = run_experiment("fit_grid",
                               recalculate_experiments=True, \
                               n_jobs = 4, \
                               experiment_detailed_name="fit_grid_checkerboard_uncertanity",
                               base_experiment="fit_active_learning",
                               seed=777,
                               grid_params = {"base_model_kwargs:alpha": list(np.logspace(-5,5,10))},
                               base_experiment_kwargs={"strategy": "uncertanity_sampling",
                                                       "loader_function": "get_splitted_uniform_data",
                                                       "batch_size": 1, \
                                                       "base_model": "SGDClassifier",
                                                       "loader_args": {"n_folds": 2}})

        grid_results_random = run_experiment("fit_grid",
                               experiment_detailed_name="fit_grid_checkerboard_random",
                               base_experiment="fit_active_learning", seed=777,
                               grid_params = {"base_model_kwargs:alpha": list(np.logspace(-5,5,10))},
                               base_experiment_kwargs={"strategy": "random_query",
                                                       "loader_function": "get_splitted_uniform_data",
                                                       "batch_size": 1,
                                                       "base_model": "SGDClassifier",
                                               "loader_args": {"n_folds": 2}})

        random_exp = get_best(grid_results_random.experiments, "mean_mcc_valid")

        uncert_exp = get_best(grid_results_uncert.experiments, "mean_mcc_valid")

        # TODO: what's good condition for being better? Probably mean integral of curve.
        # plot_monitors([uncert_exp, random_exp])
        # print np.array(random_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum()
        # print np.array(uncert_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum()
        # self.assertTrue(2*np.array(random_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum() <
        #     np.array(uncert_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum()
        # )