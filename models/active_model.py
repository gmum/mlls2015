from sklearn.base import BaseEstimator
from models.utils import ObstructedY
from models.strategy import random_query
from models.strategy import query_by_bagging
import numpy as np

class ActiveModel(BaseEstimator):

    def __init__(self, strategy, base_model):

        self.strategy = strategy
        self.base_model = base_model
        self.has_partial =  hasattr(self.base_model, 'partial_fit')
        self.y = None

    def fit(self, X, y, n_label=None, n_iter=None, strategy_args={}, fit_args={}, warm_start=True, verbose=False):
        assert isinstance(y, ObstructedY), "y needs to be passed as ObstructedY"

        self.y = y
        n_already_labeled = 0
        counter = 0

        if n_label is None and n_iter is None:
            n_label = X.shape[0]

        while True:

            if counter == 0 and warm_start:
                ind_to_label = random_query(X, self.base_model, strategy_args['batch_size'], strategy_args['seed'])
            else:
                if self.strategy.__name__ == 'query_by_bagging':
                    ind_to_label = self.strategy(X, y, self.base_model, **strategy_args)
                else:
                    ind_to_label = self.strategy(X[np.invert(y.known)], self.base_model, **strategy_args)

            y.query(ind_to_label)

            if self.has_partial:
                self.base_model.partial_fit(X[ind_to_label], y[ind_to_label], classes=y.classes, **fit_args)
            else:
                self.base_model.fit(X[self.y.known], self.y[y.known])

            n_already_labeled += len(ind_to_label)
            counter += 1

            if verbose:
                print "Iter: %i, labeled %i/%i" % (counter, n_already_labeled, n_label)

            if n_label is not None and n_label - n_already_labeled == 0:
                break
            elif n_label is not None and n_label - n_already_labeled < strategy_args['batch_size']:
                strategy_args['batch_size'] = n_label - n_already_labeled
            elif n_iter is not None and counter == n_iter:
                break

    def predict(self, X):

        return self.base_model.predict(X)
