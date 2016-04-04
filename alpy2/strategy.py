"""
Strategies for picking samples from datasets.
"""
import numpy as np
from sklearn.utils import validation as val
from abc import ABCMeta, abstractmethod
from alpy2.utils import _check_masked_labels, unmasked_indices, masked_indices

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
import logging
from itertools import combinations
import pdb
logger = logging.getLogger(__name__)


# TODO: add test for QGB Numba vs QGB Python solver
# TODO: add forbidden ids to QGB Numba
def _score_qgb_python(ids, distance_cache, base_scores, c):
    all_pairs_x, all_pairs_y = zip(*list(combinations(ids, r=2)))# zip(*product(ids, ids))
    # Product has n^2 while correct number is n * (n - 1) / 2.0
    # all_pairs = (len(ids) * (len(ids) - 1))
    # Should have this length
    all_pairs = (len(ids) * (len(ids) - 1)) / 2.0
    assert len(all_pairs_x) == all_pairs
    return (1. - c) * base_scores[ids].mean() + \
           (c / all_pairs) * distance_cache[all_pairs_x, all_pairs_y].sum()

QGB_DIST_AVG = 0
QGB_DIST_MIN = 1
QGB_DIST_GLOBAL_MIN = 2


def _qgb_solver_python(distance, base_scores, warm_start, c, batch_size, dist_fnc=QGB_DIST_AVG, forbidden_ids=[]):
    assert all(base_scores <= 1.0) and all(base_scores >= 0.0)
    picked_sequence = list(warm_start)
    picked = set(picked_sequence)
    candidates = [i for i in range(base_scores.shape[0]) if (i not in picked) and (i not in forbidden_ids)]
    distances_to_picked = np.zeros(shape=(distance.shape[0], ), dtype=np.float64)

    assert len(base_scores) == len(distance)

    # Mathematical explanation of distances_to_picked:
    # ############
    # Let's assume we are picking k-th sample to batch
    # P_ij = dist_score of j-th sample in batch IF i-th sample was to be included
    # L = dist_score of batch
    # distances_to_picked_i = \sum_j P_ij (so it includes P_ik!)
    # For QGB_DIST_AVG we have the property
    # that P_ij is constant between iterations and just P_*k column changes
    # we keep track only of \delta:
    # distance_to_picked_i = \sum_j P_ij - L
    # and we try to maximize this difference
    if dist_fnc == QGB_DIST_AVG:
        distances_to_picked[:] = distance[:, picked_sequence].sum(axis=1)

        while len(picked) < batch_size and len(candidates) > 0:
            # FIXME: uncommented all_pairs and max(1, len(picked)) is slightly incorrect
            # but is kept for backward compability
            # all_pairs = max(1, len(picked) * (len(picked) + 1) / 2.0)
            all_pairs = max(1, len(picked) * (len(picked) - 1) / 2.0)

            candidates_scores = c * distances_to_picked[candidates] / all_pairs \
                                + (1 - c) * base_scores[candidates] / max(1, len(picked) ) # + 1

            candidates_index = np.argmax(candidates_scores)
            new_index = candidates[candidates_index]
            picked.add(new_index)
            picked_sequence.append(new_index)
            del candidates[candidates_index]

            distances_to_picked += distance[:, new_index]
    else:
        assert dist_fnc == QGB_DIST_GLOBAL_MIN, "Not supported other fnc"

        for i in range(len(distance)):
            distance[i, i] = np.inf # So we never pick it as the minimum

        L = np.ones(shape=(batch_size,))
        for id, i in enumerate(picked_sequence):
            L[id] = distance[i, picked_sequence].min()

        while len(picked) < batch_size and len(candidates) > 0:
            candidates_scores = np.zeros(shape=(len(candidates,)))
            current_score = L.min()
            # Should also divide but uncessary
            # O(batch_size * N) - for each candidate looks through batch_size
            for id, i in enumerate(candidates):
                # We don't care about scales being off. We will pick C in loguniform scale
                # \delta is linear so we score candidates as delta of score it will bring
                if len(picked_sequence):
                    new_dist_score = min(current_score, np.min(distance[i, picked_sequence]))
                else:
                    new_dist_score = 0

                candidates_scores[id] = c * (new_dist_score - current_score) + \
                                      (1 - c) * (base_scores[i])

            candidates_index = np.argmax(candidates_scores)
            new_index = candidates[candidates_index]
            picked.add(new_index)
            picked_sequence.append(new_index)
            del candidates[candidates_index]

            # Update stuff TODO: could be faster
            for id, i in enumerate(picked_sequence):
                L[id] = distance[i, picked_sequence].min()

    if dist_fnc == QGB_DIST_AVG:
        # We use trick that it is enough to sum it
        picked_dissimilarity = distances_to_picked[picked_sequence].sum() / 2.0
        B = (1.0 / max(1, len(picked) * (len(picked) - 1) / 2.0)) * picked_dissimilarity
    else:
        B = L.min()

    assert 0 <= B <= 1, "Found correct B"

    A = base_scores[picked_sequence].mean()
    scores = (1 - c) * A \
             + c * B

    return picked_sequence, scores

_qgb_solver_numba = None
try:
    import numba
    from numba import autojit

    def _qgb_solver_numba(distance, base_scores, warm_start, c, batch_size):
        picked = np.zeros(shape=(batch_size, ), dtype=np.int32)
        selected = len(warm_start)
        picked[0:len(warm_start)] = warm_start

        # Candidates has in range 0:N_candidates current candidates
        candidates = np.arange(base_scores.shape[0])
        N_candidates = len(candidates)
        # Delete candidates from warm start
        for id in warm_start:
            candidates[id], candidates[N_candidates - 1] = candidates[N_candidates - 1], \
                candidates[id]
            N_candidates -= 1

        distances_to_picked = np.zeros(shape=(distance.shape[0], ), dtype=np.float32)
        candidates_scores = np.zeros(shape=(distance.shape[0],), dtype=np.float32)
        for i in range(len(warm_start)):
            distances_to_picked[:] += distance[:, warm_start[i]]#.sum(axis=1)

        assert N_candidates > 0
        while selected < batch_size and N_candidates > 0:
            all_pairs = max(1, selected * (selected + 1) / 2.0)
            # TODO: optimize - we make copy every iteration, this could be calculated as iterator

            for i in range(N_candidates):
                candidates_scores[i] = c * distances_to_picked[candidates[i]] / all_pairs \
                                + (1 - c) * base_scores[candidates[i]] / max(1, selected + 1)

            candidates_index = np.argmax(candidates_scores)

            new_index = candidates[candidates_index]
            picked[selected] = new_index
            distances_to_picked += distance[:, new_index]

            # Swap
            candidates[candidates_index], candidates[N_candidates - 1] = candidates[N_candidates - 1], \
                candidates[candidates_index]
            candidates_scores[N_candidates - 1] = 0

            N_candidates -= 1
            selected += 1

        # This stores (x_i, a_j), where x_i is from whole dataset and a_j is from picked subset
        # (a_i, a_j) and (a_j, a_i) - those are doubled
        picked_dissimilarity = distances_to_picked[picked[0:selected]].sum() / 2.0
        A = base_scores[picked[0:selected]].sum() / max(1.0, float(selected))
        B = (1.0 / max(1, selected * (selected - 1) / 2.0)) * picked_dissimilarity
        scores = (1 - c) * A \
                 + c * B

        return list(picked), scores

    _qgb_solver_numba = autojit(nopython=True)(_qgb_solver_numba)
except Exception, e:
    logger.warning("Failed numba compilation with {}, part of optimizations unavailable".format(e))



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


    def __call__(self, X, y, model, batch_size, return_score=False, **kwargs):
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

        # self._check()

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
            # Negative entropy
            fitness = np.sum(p * np.log(np.clip(p, 1e-5, 1 - 1e-5)), axis=1).ravel()
            ids = np.argsort(fitness)[:batch_size]

        else:
            raise AttributeError("Model with either `decision_function` or `predict_proba` method")

        if not return_score:
            return unknown_ids[ids]
        else:
            fitness = np.abs(fitness)
            max_fit = np.max(fitness)
            return unknown_ids[ids], (max_fit - fitness)/max_fit


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
        # self._check(X, y)

        if self.method == 'KL' and not hasattr(model, 'predict_proba'):
            raise AttributeError("Model with probability prediction needs to be passed for KL method!")

        known_ids = unmasked_indices(y)
        unknown_ids = masked_indices(y)

        if isinstance(model, GridSearchCV):
            estimator = getattr(model, "best_estimator_", None)
            assert estimator is not None
        else:
            estimator = model
        pairwise = getattr(estimator, "_pairwise", False)

        # pairwise = getattr(model, "_pairwise", False) or \
        #     getattr(getattr(model, "estimator", {}), "_pairwise", False)

        X_known = X[known_ids, :][:, known_ids] if pairwise else X[known_ids]
        X_unknown = X[unknown_ids, :][:, known_ids] if pairwise else X[unknown_ids]

        clfs = BaggingClassifier(estimator, n_estimators=self.n_estimators, random_state=rng)

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




class QuasiGreedyBatch2(BaseStrategy):
    """

    Parameters
    ----------
    distance_cache: np.ndarray
        2D Array of size [N, N] with precalculated pairwise distances

    c: float
        Parameter controlling the ratio of uncertainty and distance in scoring potential candidates

    base_strategy: callable
        Base strategy to use for uncertainty (ot other score type) scoring, default alpy2.strategy.UncertaintySampling

    n_tries: int
        How many different random initialisation to use, if `n_tries` > 1, the strategy picks the best try, measured by
        score, default 1

    optim: int
        Optimization level. Higher than 0 requires installed Cython and numba packages. Not used yet.

    Returns
    -------
    indices: numpy.ndarray
    """

    def __init__(self, distance_cache=None, c=0.3, base_strategy=UncertaintySampling(), dist_fnc="sum", n_tries=1, optim=0):

        assert distance_cache is not None, "In alpy2 distance_cache has to be passed"

        if distance_cache is not None and not isinstance(distance_cache, np.ndarray):
            raise TypeError("Please pass precalculated pairwise distance `distance_cache` as numpy.array")

        if isinstance(distance_cache, np.ndarray) and distance_cache.shape[0] != distance_cache.shape[1]:
            raise ValueError("`distance_cache` is expected to be a square 2D array")

        if not isinstance(c, float):
            raise TypeError("`c` is expected to be float in range [0,1], got {0}".format(type(c)))

        if c < 0 or c > 1:
            raise ValueError("`c` is expected to be float in range [0,1], got {0}".format(str(c)))

        if not callable(base_strategy):
            raise AttributeError("`base_strategy is expected to be callable`")

        if not isinstance(n_tries, int):
            raise TypeError('`n_tries` is expected to be positive integer')

        if n_tries <= 0:
            raise ValueError('`n_tries` is expected to be positive integer')

        # self.dist_fnc = dist_fnc
        self.c = c
        self.distance_cache = distance_cache
        self.base_strategy = base_strategy
        self.n_tries = n_tries
        self.optim = optim

        super(QuasiGreedyBatch2, self).__init__()


    def __call__(self, X, y, model, batch_size, rng, sample_first=False, return_score=False, forbidden_ids=[]):
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

        if self.distance_cache is not None:
            assert X.shape[0] == self.distance_cache.shape[0]

        # self._check(X, y)

        if self.n_tries == 1:
            return self._single_call(X=X,
                                     y=y,
                                     model=model,
                                     batch_size=batch_size,
                                     rng=rng,
                                     sample_first=sample_first,
                                     return_score=return_score,
                                     forbidden_ids=forbidden_ids)
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


    # def calculate_score(self, X, y, ids):

    # TODO: refactor to use uct_strategy calls
    def _single_call(self, X, y, model, batch_size, rng, sample_first=False, return_score=False, forbidden_ids=[]):

        unknown_ids = masked_indices(y)

        if len(unknown_ids) <= batch_size:
            if return_score:
                return unknown_ids, 0
            else:
                return unknown_ids

        pairwise = getattr(model, "_pairwise", False) or \
                   getattr(getattr(model, "estimator", {}), "_pairwise", False)
        X_unknown = X[unknown_ids, :][:, unknown_ids] if pairwise else X[unknown_ids]

        distance = self.distance_cache[unknown_ids, :][:, unknown_ids] if self.distance_cache is not None else 1 - X[unknown_ids, :][:, unknown_ids]
        assert np.max(distance) <= 1 and np.min(distance) >= 0

        # keep distance from all examples to picked set, 0 for now
        distances_to_picked = np.zeros(shape=(X_unknown.shape[0], ))

        _, base_scores = self.base_strategy(X=X, y=y, model=model, batch_size=batch_size, rng=rng, return_score=True)
        assert np.max(base_scores) <= 1

        if sample_first:
            p = base_scores / np.sum(base_scores)
            start_point = rng.choice(X_unknown.shape[0], p=p)
            picked_sequence = [start_point]
            picked = set(picked_sequence)

            distances_to_picked[:] = distance[:, picked_sequence].sum(axis=1)
        else:
            picked = set([])
            picked_sequence = []

        n_known_labels = len(unmasked_indices(y))

        forbidden_ids = set(forbidden_ids)
        candidates = [i for i, val in enumerate(unknown_ids) if i not in picked and val not in forbidden_ids]

        print distances_to_picked.dtype

        assert len(candidates) > 0

        while len(picked) < batch_size:
            # Have we exhausted all of our options?
            if n_known_labels + len(picked) == y.shape[0] or len(candidates) == 0:
                break

            all_pairs = max(1, len(picked) * (len(picked) - 1) / 2.0)

            # TODO: optimize - we make copy every iteration, this could be calculated as iterator
            candidates_scores = self.c * distances_to_picked[candidates] / all_pairs \
                                + (1 - self.c) * base_scores[candidates] / max(1, len(picked))
            print candidates_scores[candidates[0]], distances_to_picked[candidates[0]], base_scores[candidates[0]]
            # print candidates_scores[0], distances_to_picked[0], base_scores[0], self.c, all_pairs, len(picked)
            candidates_index = np.argmax(candidates_scores.reshape(-1))
            new_index = candidates[candidates_index]
            picked.add(new_index)
            picked_sequence.append(new_index)
            del candidates[candidates_index]

            distances_to_picked += distance[:, new_index]

        if not return_score:
            return [unknown_ids[i] for i in picked_sequence]
        else:
            # This stores (x_i, a_j), where x_i is from whole dataset and a_j is from picked subset
            # (a_i, a_j) and (a_j, a_i) - those are doubled
            picked_dissimilarity = distances_to_picked[picked_sequence].sum() / 2.0
            A = base_scores[picked_sequence].mean()
            B = (1.0 / max(1, len(picked) * (len(picked) - 1) / 2.0)) * picked_dissimilarity
            # logger.info("{}, {}".format(A, B))
            scores = (1 - self.c) * A \
                     + self.c * B
            return [unknown_ids[i] for i in picked_sequence], scores


class QuasiGreedyBatch(BaseStrategy):
    """

    Parameters
    ----------
    distance_cache: np.ndarray
        2D Array of size [N, N] with precalculated pairwise distances

    c: float
        Parameter controlling the ratio of uncertainty and distance in scoring potential candidates

    base_strategy: callable
        Base strategy to use for uncertainty (ot other score type) scoring, default alpy2.strategy.UncertaintySampling

    n_tries: int
        How many different random initialisation to use, if `n_tries` > 1, the strategy picks the best try, measured by
        score, default 1

    optim: int
        Optimization level. Higher than 0 requires installed Cython and numba packages. Not used yet.

    Returns
    -------
    indices: numpy.ndarray
    """

    def __init__(self, distance_cache=None, c=0.3, base_strategy=UncertaintySampling(), n_tries=1, optim=0, dist_fnc=QGB_DIST_AVG):
        assert distance_cache is not None, "In alpy2 distance_cache has to be passed"

        if distance_cache is not None and not isinstance(distance_cache, np.ndarray):
            raise TypeError("Please pass precalculated pairwise distance `distance_cache` as numpy.array")

        if isinstance(distance_cache, np.ndarray) and distance_cache.shape[0] != distance_cache.shape[1]:
            raise ValueError("`distance_cache` is expected to be a square 2D array")

        if not isinstance(c, float):
            raise TypeError("`c` is expected to be float in range [0,1], got {0}".format(type(c)))

        if c < 0 or c > 1:
            raise ValueError("`c` is expected to be float in range [0,1], got {0}".format(str(c)))

        if not callable(base_strategy):
            raise AttributeError("`base_strategy is expected to be callable`")

        if not isinstance(n_tries, int):
            raise TypeError('`n_tries` is expected to be positive integer')

        if n_tries <= 0:
            raise ValueError('`n_tries` is expected to be positive integer')

        # self.dist_fnc = dist_fnc
        self.c = c
        self.distance_cache = distance_cache
        self.base_strategy = base_strategy
        self.n_tries = n_tries
        self.optim = optim
        self.dist_fnc = dist_fnc
        super(QuasiGreedyBatch, self).__init__()


    def __call__(self, X, y, model, batch_size, rng, sample_first=False, return_score=False, forbidden_ids=[]):
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

        if self.distance_cache is not None:
            assert X.shape[0] == self.distance_cache.shape[0]

        # self._check(X, y)

        if self.n_tries == 1:
            return self._single_call(X=X,
                                     y=y,
                                     model=model,
                                     batch_size=batch_size,
                                     rng=rng,
                                     sample_first=sample_first,
                                     return_score=return_score,
                                     forbidden_ids=forbidden_ids)
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


    # def calculate_score(self, X, y, ids):

    def _single_call(self, X, y, model, batch_size, rng, sample_first=False, return_score=False, forbidden_ids=[]):

        unknown_ids = masked_indices(y)

        if len(unknown_ids) <= batch_size:
            if return_score:
                return unknown_ids, 0
            else:
                return unknown_ids

        pairwise = getattr(model, "_pairwise", False) or \
                   getattr(getattr(model, "estimator", {}), "_pairwise", False)
        X_unknown = X[unknown_ids, :][:, unknown_ids] if pairwise else X[unknown_ids]

        id_true_to_id_unknown = {id_true: id_unknown for id_unknown, id_true in enumerate(unknown_ids)}
        forbidden_ids_unknown = [id_true_to_id_unknown[id_true] for id_true in forbidden_ids if id_true in id_true_to_id_unknown]

        distance = self.distance_cache[unknown_ids, :][:, unknown_ids] if self.distance_cache is not None else 1 - X[unknown_ids, :][:, unknown_ids]
        assert np.max(distance) <= 1 and np.min(distance) >= 0

        _, base_scores = self.base_strategy(X=X, y=y, model=model, batch_size=batch_size, rng=rng, return_score=True)
        assert np.max(base_scores) <= 1, "Otherwise it can favor uncertainty"
        assert len(base_scores) == len(X_unknown), "Indexing agrees between base_scores and distance"

        if sample_first:
            p = base_scores / np.sum(base_scores)
            start_point = rng.choice(X_unknown.shape[0], p=p)
            picked_sequence = [start_point]
        else:
            picked_sequence = []

        picked_sequence, score = _qgb_solver_python(distance, base_scores, np.array(picked_sequence, dtype="int32"), self.c, batch_size,
                                             forbidden_ids=forbidden_ids_unknown, dist_fnc=self.dist_fnc)

        if not return_score:
            return [unknown_ids[i] for i in picked_sequence]
        else:
            return [unknown_ids[i] for i in picked_sequence], score



class CSJSampling(BaseStrategy):

    def __init__(self, c, projection, k=2, distance_cache=None):

        if distance_cache is not None and not isinstance(distance_cache, np.ndarray):
            raise TypeError("Please pass precalculated pairwise distance `distance_cache` as numpy.array")

        if distance_cache is not None and distance_cache.shape[0] != distance_cache.shape[1]:
            raise ValueError("`distance_cache` is expected to be a square 2D array")

        if not isinstance(c, float):
            raise TypeError("`c` is expected to ne float in range [0,1]")

        if c < 0 or c > 1:
            raise ValueError("`c` is expected to ne float in range [0,1]")

        if not isinstance(projection, np.ndarray):
            raise TypeError("Please pass precalculated `projection` as numpy.array")

        if not isinstance(k, int):
            raise TypeError('`k` is expected to be int > 2')

        if k < 2:
            raise ValueError('`k` is expected to be int > 2')

        self.distance_cache = distance_cache
        self.c = c
        # NOTE: we assume that this will always be the same projection (ie. sorensen), and that it is
        # different than the one used for validation clusters
        self.projection = projection
        self.qgb = QuasiGreedyBatch(distance_cache=distance_cache, c=c)
        self.k = k


    def __call__(self, X, y, model, batch_size, rng):

        unknown_ids = masked_indices(y)

        X_proj = self.projection

        assert X_proj.shape[0] == X.shape[0]

        if len(unknown_ids) <= batch_size:
            return unknown_ids

        # Cluster and get uncertanity
        cluster_ids = KMeans(n_clusters=self.k, random_state=rng).fit_predict(X_proj[unknown_ids])

        examples_by_cluster = {cluster_id_key:
                                   np.where(cluster_ids == cluster_id_key)[0]
                               for cluster_id_key in np.unique(cluster_ids)
                               }

        for k in examples_by_cluster:
            for ex_id in examples_by_cluster[k]:
                assert cluster_ids[ex_id] == k

        # too_small_clusters = []
        # batch_sizes = [0 for _ in range(self.k)]
        # left_to_allocate = batch_size
        # for cluster_id, cluster_examples in examples_by_cluster.iteritems():
        #     if len(cluster_examples) < batch_size / self.k:
        #         too_small_clusters.append(cluster_id)
        #         batch_sizes[cluster_id] = len(cluster_examples)
        #         left_to_allocate -= len(cluster_examples)
        #
        # assert len(too_small_clusters) < len(batch_sizes)
        # clusters_to_allocate = [cluster_id for cluster_id, allocation in enumerate(batch_sizes) if allocation == 0]
        # for cluster_id in clusters_to_allocate:
        #     if len(examples_by_cluster[cluster_id]) < left_to_allocate / len(clusters_to_allocate):
        #         batch_sizes[cluster_id] = examples_by_cluster[cluster_id]
        #     else:
        #         batch_sizes[cluster_id] = left_to_allocate / len(clusters_to_allocate)

        if len(examples_by_cluster[0]) < batch_size / 2:
            batch_sizes = [len(examples_by_cluster[0]), batch_size - len(examples_by_cluster[0])]
        elif len(examples_by_cluster[1]) < batch_size / 2:
            batch_sizes = [batch_size - len(examples_by_cluster[1]), len(examples_by_cluster[1])]
        else:
            batch_sizes = [batch_size/2, batch_size - batch_size/2]

        picked = []
        # Call quasi greedy for each cluster_id
        for id, cluster_id in enumerate(np.unique(cluster_ids)):

            forbidden_ids = []
            for cluster_id_2 in np.unique(cluster_ids):
                if cluster_id_2 != cluster_id:
                    forbidden_ids += unknown_ids[examples_by_cluster[cluster_id_2]].tolist()

            picked_cluster = self.qgb(X, y=y, model=model, rng=rng, batch_size=batch_sizes[id], forbidden_ids=forbidden_ids)

            reverse_dict = {id_true: id_rel for id_rel, id_true in enumerate(unknown_ids)}

            assert all(reverse_dict[example_id] in examples_by_cluster[cluster_id] for example_id in picked_cluster)

            picked += picked_cluster

        return picked