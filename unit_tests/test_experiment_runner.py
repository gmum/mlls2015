import sys
sys.path.append("..")

import kaggle_ninja

from experiments import random_query, random_query_composite

from experiments import experiment_runner
# from experiment_runner import _replace_in_json
from experiment_runner import *

import unittest
import os

class TestDataAPI(unittest.TestCase):

    def setUp(self):
        self.comps = [['5ht7', 'ExtFP']]
        self.n_folds = 3

    def test_composite_experiment(self):
        turn_on_force_reload_all()
        # grid = {"C": [10,20,30], "dataset": {"path": 20}, "values": [1,2]}
        # grid2 = {}
        # self.assertTrue(_replace_in_json(grid, "dataset.path", 10)[0]['dataset']['path'] == 10)
        # self.assertTrue(_replace_in_json(grid, "values.1", 10)[0]['values'][1] == 10)
        run_experiment(random_query.ex, batch_size=10, seed=655)
        run_experiment(random_query.ex, batch_size=20, seed=655)
        turn_off_force_reload_all()
        run_experiment(random_query_composite.ex, base_batch_size=10, seed=655)


    def test_grid_params_setting(self):
        os.system("rm "+os.path.join(c["CACHE_DIR"], "_key_storage_random_query*"))
        turn_on_force_reload_all()
        # grid = {"C": [10,20,30], "dataset": {"path": 20}, "values": [1,2]}
        # grid2 = {}
        # self.assertTrue(_replace_in_json(grid, "dataset.path", 10)[0]['dataset']['path'] == 10)
        # self.assertTrue(_replace_in_json(grid, "values.1", 10)[0]['values'][1] == 10)
        results = run_experiment_grid(name="random_query", n_jobs=4, recalculate=True, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        # Test repeatability
        self.assertTrue(any(abs(results[i].results['acc'] - 0.65814696485623003)<1e-2) for i in range(4))
        self.assertTrue(os.system("ls "+os.path.join(c["CACHE_DIR"], "_key_storage_random_query*")) == 0)
        turn_off_force_reload_all()
        results = run_experiment_grid(name="random_query", n_jobs=4, timeout=20, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        # Test repeatability
        self.assertTrue(any(abs(results[i].results['acc'] - 0.65814696485623003)<1e-2) for i in range(4))

        results = run_experiment_grid(name="random_query", n_jobs=4, timeout=1, grid_params={"batch_size": [10,20,30,40]}, seed=777)
        self.assertTrue(all(isinstance(r, str) for r in results))



