import sys
sys.path.append("..")
import kaggle_ninja
from kaggle_ninja import *
import random_query, random_query_composite
from experiments import experiment_runner, al_simulation
from experiment_runner import run_experiment, run_experiment_grid
from misc.config import *
# from experiment_runner import _replace_in_json


import unittest
import os

class TestDataAPI(unittest.TestCase):

    def test_basic_caching_al_simulation(self):
        r1 = run_experiment("al_simulation", experiment_name="random_query", \
                                 strategy="random_query", loader_args={"n_folds": 10})

        import time
        start = time.time()
        r2 = run_experiment("al_simulation", experiment_name="random_query",
                            strategy="random_query", loader_args={"n_folds": 10})

        # It cached if calculated result in less than 1s
        # Note: this is indeterministic test, so if it fails and is close to 1s you can adjust.
        self.assertLess(time.time() - start, 1)

    def test_composite_experiment(self):
        turn_on_force_reload_all()
        run_experiment("random_query_exp", loader_args={"n_folds":2}, batch_size=10, seed=655)
        results = run_experiment("random_query_exp", loader_args={"n_folds":2},  batch_size=20, seed=655)
        # Number of evaluated folds is correct
        self.assertTrue(len(results.results['mcc_valid'])==2)
        turn_off_force_reload_all()
        run_experiment("random_query_composite", base_batch_size=10, seed=655)


    def test_grid_params_setting(self):
        os.system("rm "+os.path.join(c["CACHE_DIR"], "_key_storage_random_query*"))
        turn_on_force_reload_all()
        results = run_experiment_grid(name="random_query_exp", loader_args={"n_folds":2},  n_jobs=4, \
                                      recalculate=True, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        # Test repeatability
        before = (sum(r.results['mean_mcc_valid'] for r in results))

        self.assertTrue(os.system("ls "+os.path.join(c["CACHE_DIR"], "_key_storage_random_query*")) == 0)
        turn_off_force_reload_all()
        results = run_experiment_grid(name="random_query_exp", n_jobs=4, loader_args={"n_folds":2}, \
                                      timeout=20, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        after = (sum(r.results['mean_mcc_valid'] for r in results))

        self.assertAlmostEqual(before, after)

        # Test timeouting
        results = run_experiment_grid(name="random_query_exp", n_jobs=4, timeout=1, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        self.assertTrue(all(isinstance(r, str) for r in results))

