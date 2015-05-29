
import sys
sys.path.append("..")


import kaggle_ninja
kaggle_ninja.turn_on_force_reload_all()

import unittest
from get_data import get_data, get_splitted_data, bucket_simple_threshold

class TestDataAPI(unittest.TestCase):

    def setUp(self):
        self.comps = [['5ht7', 'ExtFP']]
        self.n_folds = 3

    def test_splitted_data(self):
        loader = ["get_splitted_data",
                  {"n_folds": self.n_folds,
                   "seed":777,
                   "test_size":0.0}]
        preprocess_fncs = []

        data = get_data(self.comps, loader, preprocess_fncs)
        print data.values()[0][1]
        folds = data.values()[0][0]
        test_data = data.values()[0][1]
        data_desc = data.values()[0][2]

        self.assertEqual(len(data), 1)
        self.assertEqual(len(test_data), 0)
        self.assertEqual(len(folds), self.n_folds)
        self.assertEqual(len(data_desc.values()), 2)

    def test_spliting_with_test_data(self):
        loader = ["get_splitted_data",
                  {"n_folds": self.n_folds,
                   "seed":777,
                   "test_size":0.2}]
        preprocess_fncs = []

        data = get_data(self.comps, loader, preprocess_fncs)
        X_test, y_test = data.values()[0][1]
        folds = data.values()[0][0]

        self.assertTrue(X_test.shape[0] > 0)
        self.assertTrue(X_test.shape[0] == y_test.shape[0])
        self.assertTrue(4 * X_test.shape[0] -  (folds[0]['X_train'].shape[0] +folds[0]['X_valid'].shape[0] ) < 10)

    def test_bucketing(self):
        loader = ["get_splitted_data",
                  {"n_folds": 3,
                   "seed":777,
                   "test_size":0.0}]

        preprocess_fncs = [["to_binary", {"all_below": True}]]
        data = get_data(self.comps, loader, preprocess_fncs)
        folds = data.values()[0][0]

        self.assertTrue(folds[0]['X_train'].shape[1] != folds[1]['X_train'].shape[1])

    def test_bucketing_with_test_data(self):
        loader = ["get_splitted_data",
                  {"n_folds": 2,
                   "seed":777,
                   "test_size":0.2}]

        preprocess_fncs = [["to_binary", {"all_below": True}]]
        data = get_data(self.comps, loader, preprocess_fncs)
        folds = data.values()[0][0]
        X_test, y_test = data.values()[0][1]

        self.assertEqual(folds[0]['X_train'].shape[1], X_test.shape[1])

suite = unittest.TestLoader().loadTestsFromTestCase(TestDataAPI)
print unittest.TextTestRunner(verbosity=3).run(suite)
