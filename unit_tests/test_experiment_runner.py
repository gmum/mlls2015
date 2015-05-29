import sys
sys.path.append("..")

import kaggle_ninja

from experiments import experiment_runner

import unittest

class TestDataAPI(unittest.TestCase):

    def setUp(self):
        self.comps = [['5ht7', 'ExtFP']]
        self.n_folds = 3

    def test_splitted_data(self):
        kaggle_ninja.turn_on_force_reload_all()
        experiment_runner.run_experiment(name="random_query", batch_size=10)
        kaggle_ninja.turn_off_force_reload_all()

        result = experiment_runner.run_experiment(_load_cache_or_fail=True, name="random_query", batch_size=10)
        self.assertTrue(result is not None)

        result = experiment_runner.run_experiment(_load_cache_or_fail=True, name="random_query", batch_size=20)
        self.assertTrue(result is None)