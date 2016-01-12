"""
Strategies for picking samples from datasets.
"""
import numpy as np
from sklearn.utils import validation as val
from abc import ABCMeta, abstractmethod
from alpy.utils import _check_masked_labels, unmasked_indices, masked_indices
from sklearn.ensemble import BaggingClassifier



class BaseStrategy(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X, y, model, batch_size):
        raise NotImplemented("Virtual method got called")

    def _check(self, X, y):

        _check_masked_labels(y)
        X = val.as_float_array(X)
        val.check_consistent_length(X, y)


class UncertaintySampling(BaseStrategy):

    def __init__(self):
        super(UncertaintySampling, self).__init__()


    def __call__(self, X, y, model, batch_size, **kwargs):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.
        y: numpy.ma.masked_array
            Masked numpy array with shape (n, )
            The mask is masking with True the unknown labels in `y`.
        current_model: sklearn estimator
        batch_size: int
            How many examples to query in one active learning loop iteration.
        Returns
        -------
        indices: numpy.ndarray
        """

        self._check()

        unknown_ids = masked_indices(y)
        known_ids = unmasked_indices(y)

        # mask samples with obscure labels
        pairwise = getattr(model, "_pairwise", False) or \
                   getattr(getattr(model, "estimator", {}), "_pairwise", False)
        X = X[unknown_ids, :][:, known_ids] if pairwise else X[unknown_ids]


        if hasattr(model, "decision_function"):
            # Settles page 12
            fitness = np.abs(np.ravel(model.decision_function(X)))
            ids = np.argsort(fitness)[:batch_size]

        elif hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # Settles page 13
            fitness = np.sum(p * np.log(p), axis=1).ravel()
            ids = np.argsort(fitness)[:batch_size]

        else:
            raise AttributeError("Model with either `decision_function` or `predict_proba` method")

        return unknown_ids[ids]


class PassiveStrategy(BaseStrategy):

    def __init__(self):
        super(PassiveStrategy, self).__init__()

    def __call__(self, X, y, model, batch_size, **kwargs):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.
        batch_size: int
            How many examples to query in one active learning loop iteration.
        rng:
            seed for random number generator
        Returns
        -------
        indices: numpy.ndarray
        """

        rng = val.check_random_state(kwargs['rng'])
        unknown_ids = masked_indices(y)
        selected = rng.choice(len(unknown_ids), size=min(batch_size, len(unknown_ids)), replace=False)
        return unknown_ids[selected]


class QueryByBagging(BaseStrategy):
    """
    Parameters
    ----------
    n_estimators: int
        How many estimators to use in constructing the committee

    method: string
        Available are `KL` and `entropy`, `KL` requires model passed to call to have `predict_proba` attribute

    Returns
    -------
    Callable QueryByBagginf strategy instance

    """

    def __init__(self, n_estimators=5, method="KL"):

        if method not in ["KL", "entropy"]:
            raise ValueError("`method` is suppose to be `KL` or `entropy`")
        self.method = method

        self.n_estimators = n_estimators
        self.eps = 1e-6

        super(QueryByBagging, self).__init__()

    def __call__(self, X, y, model, batch_size, rng, **kwargs):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.

        y: numpy.ma.masked_array
            Masked numpy array with shape (n, )
            The mask is masking with True the unknown labels in `y`.

        current_model: sklearn estimator

        batch_size: int
            How many examples to query in one active learning loop iteration.

        rng: numpy.random.RandomState
            RandomState for BaggingClassifier

        Returns
        -------
        indices: numpy.ndarray

        """
        # check
        self._check(X, y)

        if self.method == 'KL' and not hasattr(model, 'predict_proba'):
            raise AttributeError("Model with probability prediction needs to be passed for KL method!")

        known_ids = unmasked_indices(y)
        unknown_ids = masked_indices(y)

        pairwise = getattr(model, "_pairwise", False) or \
            getattr(getattr(model, "estimator", {}), "_pairwise", False)
        X_known = X[known_ids, :][:, known_ids] if pairwise else X[known_ids]
        X_unknown = X[unknown_ids, :][:, unknown_ids] if pairwise else X[unknown_ids]

        clfs = BaggingClassifier(model, n_estimators=self.n_estimators, random_state=rng)

        clfs.fit(X_known, y[known_ids].data)
        pc = clfs.predict_proba(X_unknown)

        if self.method == 'entropy':
            pc += self.eps
            fitness = np.sum(pc * np.log(pc), axis=1)
            ids =  np.argsort(fitness)[:batch_size]
        elif self.method == 'KL':
            p = np.array([clf.predict_proba(X_unknown) for clf in clfs.estimators_])
            fitness = np.mean(np.sum(p * np.log(p / pc), axis=2), axis=0)
            ids = np.argsort(fitness)[-batch_size:]

        return unknown_ids[ids]


class QuasiGreedyBatch(BaseStrategy):
    """

    Parameters
    ----------
    distance_cache: np.ndarray
        2D Array of size [N, N] with precalculated pairwise distances

    c: float
        Parameter controlling the ratio of uncertainty and distance in scoring potential candidates

    base_strategy: callable
        Base strategy to use for uncertainty (ot other score type) scoring, default alpy.strategy.UncertaintySampling

    n_tries: int
        How many different random initialisation to use, if `n_tries` > 1, the strategy picks the best try, measured by
        score, default 1

    Returns
    -------
    indices: numpy.ndarray
    """

    def __init__(self, distance_cache, c=0.3, base_strategy=UncertaintySampling(), n_tries=1):

        if not isinstance(distance_cache, np.ndarray):
            raise TypeError("Please pass precalculated pairwise distance `distance_cache` as numpy.array")

        if distance_cache.shape[0] != distance_cache.shape[1]:
            raise ValueError("`distance_cache` is expected to be a square 2D array")

        if not isinstance(c, float):
            raise TypeError("`c` is expected to ne float in range [0,1]")

        if c < 0 or c > 1:
            raise ValueError("`c` is expected to ne float in range [0,1]")

        if not callable(base_strategy):
            raise AttributeError("`base_strategy is expected to be callable`")

        if not isinstance(n_tries, int):
            raise TypeError('`n_tries` is expected to be positive integer')

        if n_tries <= 0:
            raise ValueError('`n_tries` is expected to be positive integer')

        self.c = c
        self.distance_cache = distance_cache
        self.base_strategy = base_strategy
        self.n_tries = n_tries

        super(QuasiGreedyBatch, self).__init__()

    def __call__(self, X, y, model, batch_size, rng, return_score=False):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Numpy array with shape (n, m), where `n` is the number of samples
            and `m` the number of features.

        y: numpy.ma.masked_array
            Masked numpy array with shape (n, )
            The mask is masking with True the unknown labels in `y`.

        model: sklearn estimator

        batch_size: int
            How many examples to query in one active learning loop iteration.

        rng: numpy.random.RandomState
            RandomState for BaggingClassifier

        Returns
        -------
        indices: numpy.ndarray
        """

        assert X.shape[0] == self.distance_cache.shape[0]

        self._check(X, y)

        if self.n_tries == 1:
            return self._single_call(X=X,
                                     y=y,
                                     model=model,
                                     batch_size=batch_size,
                                     rng=rng,
                                     return_score=return_score)
        else:
            results = [self._single_call(X=X,
                                         y=y,
                                         model=model,
                                         batch_size=batch_size,
                                         rng=rng,
                                         sample_first=False,
                                         return_score=True)]

            for i in range(self.n_tries - 1):
                results.append(self._single_call(X=X,
                                                 y=y,
                                                 model=model,
                                                 batch_size=batch_size,
                                                 rng=rng,
                                                 sample_first=True,
                                                 return_score=True))

        if not return_score:
            return results[np.argmax([r[1] for r in results])][0]
        else:
            return results[np.argmax([r[1] for r in results])]


    def _single_call(self, X, y, model, batch_size, rng, sample_first=False, return_score=False):

        unknown_ids = masked_indices(y)

        pairwise = getattr(model, "_pairwise", False) or \
                   getattr(getattr(model, "estimator", {}), "_pairwise", False)
        X_unknown = X[unknown_ids, :][:, unknown_ids] if pairwise else X[unknown_ids]

        distance = self.distance_cache[unknown_ids, :][:, unknown_ids]

        # keep distance from all examples to picked set, 0 for now
        distatance_to_picked = np.zeros(shape=(X_unknown.shape[0], ))

        _, base_scores = self.base_strategy(X=X, y=y, model=model, batch_size=batch_size, rng=rng, return_score=True)

        if sample_first:
            p = base_scores / np.sum(base_scores)
            start_point = rng.choice(X_unknown.shape[0], p=p)
            picked_sequence = [start_point]
            picked = set(picked_sequence)
            distatance_to_picked[:] = distance[:, picked_sequence].sum(axis=1)
        else:
            picked = set([])
            picked_sequence = []

        n_known_labels = len(unmasked_indices(y))

        candidates = [i for i in range(X_unknown.shape[0]) if i not in picked]
        while len(picked) < batch_size:
            # Have we exhausted all of our options?
            if n_known_labels + len(picked) == y.shape[0]:
                break

            all_pairs = max(1, len(picked) * (len(picked) + 1) / 2.0)
            candidates_scores = self.c * distatance_to_picked[candidates] / all_pairs \
                                + (1 - self.c) * base_scores[candidates] / max(1, len(picked))
            candidates_index = np.argmax(candidates_scores.reshape(-1))
            new_index = candidates[candidates_index]
            picked.add(new_index)
            picked_sequence.append(new_index)
            del candidates[candidates_index]
            distatance_to_picked += distance[:, new_index]

        picked_dissimilarity = distatance_to_picked[picked_sequence].sum() / 2.0
        scores = (1 - self.c) * base_scores[picked_sequence].mean() \
                 + self.c * (1.0 / max(1, len(picked) * (len(picked) - 1) / 2.0)) * picked_dissimilarity

        if not return_score:
            return [unknown_ids[i] for i in picked_sequence]
        else:
            return [unknown_ids[i] for i in picked_sequence], scores


