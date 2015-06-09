from collections import defaultdict
import copy, math, sys
import numpy as np
from collections import namedtuple
import copy
import sys
sys.path.append("..")
from misc.config import *
log = main_logger
from get_data import *
from models.strategy import *
import sys
sys.path.append("..")
import unittest
import numpy as np
from models.balanced_models import *

class TestObstructedY(unittest.TestCase):

    def test_balanced_models(self):
        protein = "5ht7"
        fingerprint = "ExtFP"
        loader = ["get_splitted_data",
                  {"n_folds": 3,
                   "seed":777,
                   "test_size":0.0,
                    "compound": protein,
                      "fingerprint": fingerprint
                  }]

        preprocess_fncs = [["to_binary", {"all_below": True}]]
        X = get_data_by_name(loader, preprocess_fncs, "X_train.0")
        Y = get_data_by_name(loader, preprocess_fncs, "Y_train.0")
        X_valid = get_data_by_name(loader, preprocess_fncs, "X_train.0")
        Y_valid = get_data_by_name(loader, preprocess_fncs, "Y_train.0")

        scores_after = defaultdict(list)
        for param in [100,200,1000]:
            for model_cls in [EEM, TWELM, SVMTAN, RandomNB]:
                print model_cls
                if "SVMTAN" in model_cls.__name__:
                    m = model_cls(C=param, random_state=777).fit(X["data"], Y["data"])
                else:
                    m = model_cls(h=param,  projector=RandomProjector(), random_state=777).fit(X["data"], Y["data"])
                scores_after[model_cls.__name__].append(wac_score(Y_valid["data"], m.predict(X_valid["data"])))

        # Checks if refactor doesn't change scores
        scores_wojtek = {
            "EEM": [0.8283487744589316, 0.8902390701360454, 0.9686821895578994],
            "RandomNB": [0.7752844500632111, 0.8184095326150537, 0.82432496179029],
            "SVMTAN": [0.9956554144574221, 0.9967215125384455, 0.9968017057569296],
            "TWELM": [0.8663839462611092, 0.9247551748212163, 0.9968017057569296]
        }
        for k1,k2 in zip(sorted(scores_wojtek.keys()), sorted(scores_after.keys())):
            assert np.array_equal(scores_wojtek[k1], scores_after[k2])


        # Test with fixed projection
        f = FixedProjector(rng=777, h_max=1000, projector=RandomProjector(), X=X["data"])
        scores_after = defaultdict(list)
        for param in [100,200,1000]:
            for model_cls in [EEM, TWELM, SVMTAN, RandomNB]:
                print model_cls
                if "SVMTAN" in model_cls.__name__:
                    m = model_cls(C=param, random_state=777).fit(X["data"], Y["data"])
                else:
                    m = model_cls(h=param,  projector=f, random_state=777).fit(X["data"], Y["data"])
                scores_after[model_cls.__name__].append(wac_score(Y_valid["data"], m.predict(X_valid["data"])))

        for k1,k2 in zip(sorted(scores_wojtek.keys()), sorted(scores_after.keys())):
            print scores_wojtek[k1]
            print scores_after[k2]

        # Test that fixed random projection yields expected results
        f = FixedProjector(rng=777, h=10, h_max=1000, projector=RandomProjector(), X=X["data"])
        X_projected = RandomProjector(rng=777, h=1000).fit(X["data"]).transform(X["data"])
        assert np.array_equal(f.transform(X["data"][[50,100,12]]), X_projected[[50,100,12],0:10])


if __name__ == "__main__":
    unittest.main()