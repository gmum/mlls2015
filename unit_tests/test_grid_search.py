import unittest
import numpy as np
from get_data import get_data
import kaggle_ninja

class TestGridSearch(unittest.TestCase):

    def setUp(self):

        comps = [['5ht7', 'ExtFP']]

        loader = ["get_splitted_data", {
                "seed": 666,
                "valid_size": 0.25,
                "n_folds": 1}]

        preprocess_fncs = []

        data = get_data(comps, loader, preprocess_fncs).values()[0][0][0]
        self.X = data['X_train']['data']
        self.y = data['Y_train']['data']

        self.X_test = data['X_valid']['data']
        self.y_test = data['Y_valid']['data']

    def test_data(self):
        print self.X_test.shape
        print self.y_test.shape

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()


