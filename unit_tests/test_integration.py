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
from experiments.utils import *
from experiment_runner import run_experiment

from sklearn.decomposition import RandomizedPCA
from get_data import *



class TestDataAPI(unittest.TestCase):

    def test_clusterwise(self):
        compound = "5ht6"
        fingerprint = "ExtFP"
        seed = 777

        # all_combinations = [p for p in list(product(proteins, fingerprints))]
        preprocess_fncs = [["to_binary", {"all_below": True}]]
        loader = ["get_splitted_data_clusterwise", {
                "seed": seed,
                "valid_size": 0.15,
                "n_folds": 4}]

        folds, _, _ = get_data([[compound, fingerprint]], loader, preprocess_fncs).values()[0]

        plt.figure(figsize=(20,20))
        X_2 = folds[0]["X_valid"]
        X = folds[0]["X_train"]
        Y = folds[0]["Y_train"]["data"]

        # Note: this test might fail if you change get_data preprocess to_binary. Just change it appropr. then
        assert X["data"].shape[1] == 2012

        # Check interestigness index
        d1 = calculate_jaccard_kernel(X["data"][X["cluster_A"]], X["data"][X["cluster_A"]])[1].mean()
        d2 = calculate_jaccard_kernel(X["data"][X["cluster_B"]], X["data"][X["cluster_B"]])[1].mean()
        d3 = calculate_jaccard_kernel(X["data"][X["cluster_A"]], X["data"][X["cluster_B"]])[1].mean()
        assert d3/(0.5*(d1+d2)) >= 1.1

        ids = X["cluster_A"] + X["cluster_B"]
        c = Y.copy()[ids]
        c[:] = 1
        c[0:len(X["cluster_A"])] = 2

        X_proj = RandomizedPCA(n_components=3, iterated_power=10).fit_transform(X["data"].toarray())

        plt.figure(figsize=(30,30))
        plt.scatter(X_proj[ids,0], X_proj[ids,1], c=c, s=250)

        plt.show()

        compound = "5ht6"
        fingerprint = "ExtFP"
        seed = 777

        # all_combinations = [p for p in list(product(proteins, fingerprints))]
        preprocess_fncs = [["to_binary", {"all_below": True}]]
        loader = ["get_splitted_data_clusterwise", {
                "seed": seed,
                "valid_size": 0.15,
                "n_folds": 4}]


        twelm_uncertain_1 = run_experiment("fit_grid",
                                         n_jobs=1,
                                         experiment_detailed_name="test_fit_TWELM_uncertain_%s_%s" % (compound, fingerprint),
                                         base_experiment="fit_active_learning",
                                         seed=777,
                                         base_experiment_kwargs={"strategy": "uncertainty_sampling",
                                                                 "preprocess_fncs": preprocess_fncs,
                                                                 "batch_size": 20,
                                                                 "protein": compound,
                                                                 "fingerprint": fingerprint,
                                                                 "warm_start_percentage": 0.03,
                                                                 "base_model": "TWELM",
                                                                 "loader_function": loader[0],
                                                                 "loader_args": loader[1],
                                                                 "param_grid": {'C': list(np.logspace(-3,4,7))}})

        assert "wac_score_cluster_B_valid" in twelm_uncertain_1.experiments[0].monitors[0].keys()

        # This is rather magic, but seems quite reasonable
        assert np.array([m["wac_score_cluster_B_valid"][-1] for m in twelm_uncertain_1.experiments[0].monitors]).mean() > 0.7

    def test_reproducibility(self):
        compound = "5ht6"
        fingerprint = "ExtFP"
        seed = 777

        twelm_uncertain_1 = run_experiment("fit_grid",
                                         recalculate_experiments=True,
                                         n_jobs=8,
                                         experiment_detailed_name="test_fit_TWELM_uncertain_%s_%s" % (compound, fingerprint),
                                         base_experiment="fit_active_learning",
                                         seed=777,
                                         base_experiment_kwargs={"strategy": "uncertainty_sampling",
                                                                 "loader_function": "get_splitted_data",
                                                                 "batch_size": 20,
                                                                 "protein": compound,
                                                                 "fingerprint": fingerprint,
                                                                 "preprocess_fncs": [["to_binary", {"all_below": True}]],
                                                                 "base_model": "TWELM",
                                                                 "loader_args": {"n_folds": 2,
                                                                                 "seed": 777},
                                                                 "param_grid": {'C': list(np.logspace(-3,4,7))}})


        twelm_uncertain_2 = run_experiment("fit_grid",
                                         recalculate_experiments=True,
                                         n_jobs=8,
                                         experiment_detailed_name="test_fit_TWELM_uncertain_%s_%s" % (compound, fingerprint),
                                         base_experiment="fit_active_learning",
                                         seed=777,
                                         base_experiment_kwargs={"strategy": "uncertainty_sampling",
                                                                 "loader_function": "get_splitted_data",
                                                                 "batch_size": 20,
                                                                 "protein": compound,
                                                                 "fingerprint": fingerprint,
                                                                 "preprocess_fncs": [["to_binary", {"all_below": True}]],
                                                                 "base_model": "TWELM",
                                                                 "loader_args": {"n_folds": 2,
                                                                                 "seed": 777},
                                                                 "param_grid": {'C': list(np.logspace(-3,4,7))}})


        assert np.array_equal(twelm_uncertain_1.experiments[0].results.values(),
                              twelm_uncertain_2.experiments[0].results.values())

    # def test_checkerboard(self):
    #     grid_results_uncert = run_experiment("fit_grid",
    #                            recalculate_experiments=True, \
    #                            n_jobs = 4, \
    #                            experiment_detailed_name="test_fit_grid_checkerboard_uncertanity",
    #                            base_experiment="fit_active_learning",
    #                            seed=777,
    #                            grid_params = {"base_model_kwargs:alpha": list(np.logspace(-5,5,10))},
    #                            base_experiment_kwargs={"strategy": "uncertainty_sampling",
    #                                                    "loader_function": "get_splitted_uniform_data",
    #                                                    "preprocess_fncs": [],
    #                                                    "protein": "5ht7",
    #                                                    "fingerprint": "ExtFP",
    #                                                    "batch_size": 20, \
    #                                                    "base_model": "SGDClassifier",
    #                                                    "loader_args": {"n_folds": 2, "valid_size": 0.05}})
    #
    #     grid_results_random = run_experiment("fit_grid",
    #                            n_folds=1,
    #                            experiment_detailed_name="test_fit_grid_checkerboard_random",
    #                            base_experiment="fit_active_learning", seed=777,
    #                            grid_params = {"base_model_kwargs:alpha": list(np.logspace(-5,5,10))},
    #                            base_experiment_kwargs={"strategy": "uncertainty_sampling",
    #                                                    "loader_function": "get_splitted_uniform_data",
    #                                                    "preprocess_fncs": [],
    #                                                    "protein": "5ht7",
    #                                                    "fingerprint": "ExtFP",
    #                                                    "batch_size": 20, \
    #                                                    "base_model": "SGDClassifier",
    #                                                    "loader_args": {"n_folds": 2, "valid_size": 0.05}})
    #
    #     random_exp = get_best(grid_results_random.experiments, "mean_mcc_valid")
    #
    #     uncert_exp = get_best(grid_results_uncert.experiments, "mean_mcc_valid")
    #
    #     # TODO: what's good condition for being better? Probably mean integral of curve.
    #     # plot_monitors([uncert_exp, random_exp])
    #     # print np.array(random_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum()
    #     # print np.array(uncert_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum()
    #     # self.assertTrue(2*np.array(random_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum() <
    #     #     np.array(uncert_exp.monitors[0]["matthews_corrcoef_unlabeled"])[0:10].sum()
    #     # )