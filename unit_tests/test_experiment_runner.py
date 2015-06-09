import sys
sys.path.append("..")
import kaggle_ninja
from kaggle_ninja import *
import random_query, random_query_composite
from experiments import experiment_runner, fit_active_learning, fit_grid
from experiment_runner import run_experiment, run_experiment_grid
from experiments.utils import plot_grid_experiment_results, get_best
from misc.config import *
# from experiment_runner import _replace_in_json


import unittest
import os


class TestDataAPI(unittest.TestCase):
    def setUp(self):
        self.protein = "5ht7"
        self.fingerprint = "ExtFP"
        self.n_folds = 3
        self.preprocess_fncs  = [["to_binary", {"all_below": True}]]
        self.seed = 777

    def test_basic_caching_fit_active_learning(self):
        r1 = run_experiment("fit_active_learning",
                                n_jobs=4,
                                # force_reload=True, \
                                seed=self.seed,
                                protein=self.protein,
                                loader_function="get_splitted_data",
                                fingerprint=self.fingerprint,
                                preprocess_fncs=self.preprocess_fncs,
                                experiment_detailed_name="random_query", \
                                strategy="random_query",
                                loader_args={"n_folds": 2,
                                             "valid_size": 0.1})

        import time
        start = time.time()
        r2 = run_experiment("fit_active_learning",
                                seed=self.seed,
                                n_jobs=4,
                                protein=self.protein,
                                loader_function="get_splitted_data",
                                fingerprint=self.fingerprint,
                                preprocess_fncs=self.preprocess_fncs,
                                experiment_detailed_name="random_query", \
                                strategy="random_query",
                                loader_args={"n_folds": 2,
                                             "valid_size": 0.1})

        # It cached if calculated result in less than 1s
        # Note: this is indeterministic test, so if it fails and is close to 1s you can adjust.
        print time.time() - start
        self.assertLess(time.time() - start, 1)


    def test_grid_more_complex(self):
        # Note: this test might take a while first time

        grid_results = run_experiment("fit_grid",
                       n_jobs=4,
                       experiment_detailed_name="fit_grid_random_query",
                       base_experiment="fit_active_learning", seed=777,
                       grid_params = {"base_model_kwargs:alpha": list(np.logspace(-5,5,10)), "batch_size": [10,20]},
                       base_experiment_kwargs={"strategy": "random_query",
                                               "base_model": "SGDClassifier",
                                               "preprocess_fncs": self.preprocess_fncs,
                                               "protein": self.protein,
                                               "fingerprint": self.fingerprint,
                                               "loader_function": "get_splitted_data",
                                               "loader_args": {"n_folds": 2, "valid_size": 0.5}})


        plot_grid_experiment_results(grid_results, params=['base_model_kwargs:alpha', \
                                                           'batch_size'], metrics=['mcc_valid'])

        metric_val = get_best(grid_results.experiments, "mcc_valid").results['mcc_valid']

        c = get_best(grid_results.experiments, "mcc_valid").config
        c['force_reload'] = True

        metric_val_refit = run_experiment("fit_active_learning",\
               **c).results['mcc_valid']

        self.assertAlmostEqual(metric_val, metric_val_refit)

    def test_grid(self):
        grid_results = run_experiment("fit_grid",
               experiment_detailed_name="fit_grid_random_query",
               base_experiment="fit_active_learning", seed=777,
               grid_params = {"base_model_kwargs:alpha": [1e-1, 1]},
               base_experiment_kwargs={"strategy": "random_query",
                                       "preprocess_fncs": self.preprocess_fncs,
                                       "protein": self.protein,
                                       "fingerprint": self.fingerprint,
                                       "loader_function": "get_splitted_data",
                                       "base_model": "SGDClassifier",
                                       "loader_args": {"n_folds": 2, "valid_size":0.5}})

        self.assertTrue(grid_results.experiments[1].config['base_model_kwargs']['alpha'] == 1.0)
        self.assertTrue(grid_results.experiments[0].config['base_model_kwargs']['alpha'] == 0.1)
        self.assertTrue(grid_results.experiments[0].results['mcc_valid'] !=  \
                grid_results.experiments[1].results['mcc_valid'])


    # TODO: test:reprodcueresult from grid


    def test_composite_experiment(self):
        turn_on_force_reload_all()
        run_experiment("random_query_exp", loader_args={"n_folds":2, "valid_size": 0.5}, batch_size=10, seed=655)
        results = run_experiment("random_query_exp", loader_args={"n_folds":2, "valid_size": 0.5},  batch_size=20, seed=655)
        print results
        print results.results['mcc_valid']
        # Number of evaluated folds is correct
        self.assertTrue(len(results.results['mcc_valid'])==2)
        turn_off_force_reload_all()
        run_experiment("random_query_composite", base_batch_size=10, seed=655
                       , loader_args={"n_folds":2, "valid_size": 0.5},  batch_size=10)


    def test_grid_params_setting(self):
        os.system("rm "+os.path.join(c["CACHE_DIR"], "_key_storage_random_query*"))
        turn_on_force_reload_all()
        results = run_experiment_grid(name="random_query_exp", loader_args={"n_folds":2, "valid_size": 0.5},  n_jobs=4, \
                                      recalculate=True, grid_params={"batch_size": [10,20,30,40]}, seed=777)


        # Test repeatability
        before = (sum(r.results['mean_mcc_valid'] for r in results))

        self.assertTrue(os.system("ls "+os.path.join(c["CACHE_DIR"], "_key_storage_random_query*")) == 0)
        turn_off_force_reload_all()
        results = run_experiment_grid(name="random_query_exp", n_jobs=4, loader_args={"n_folds":2, "valid_size": 0.5}, \
                                      timeout=20, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        print results
        after = (sum(r.results['mean_mcc_valid'] for r in results))

        self.assertAlmostEqual(before, after)

        # Test timeouting
        results = run_experiment_grid(name="random_query_exp", n_jobs=4, timeout=1,
                                      loader_args={"n_folds":2, "valid_size": 0.5},

                                      grid_params={"batch_size": [10,20,30,40]}, seed=777)
        self.assertTrue(all(isinstance(r, str) for r in results))


if __name__ == "__main__":
    unittest.main()