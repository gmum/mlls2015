import sys
from sklearn.utils import check_random_state
import sys
from misc.utils import *
from models.uct import *
from alpy2.utils import masked_indices
from alpy2.strategy import *
from sklearn.metrics import pairwise_distances
from copy import deepcopy
import logging
from itertools import combinations
try:
    from cython_routines import score_cython
except ImportError:
    print "Warning: not compiled cython_routines. You can compile by running python setup.py build_ext --inplace"

logger = logging.getLogger(__name__)

def _qgb_solver_python(distance, base_scores, warm_start, c, batch_size):
    assert all(base_scores <= 1.0) and all(base_scores >= 0.0)
    picked_sequence = list(warm_start)
    picked = set(picked_sequence)
    candidates = [i for i in range(base_scores.shape[0]) if i not in picked]
    distances_to_picked = np.zeros(shape=(distance.shape[0], ), dtype=np.float32)
    distances_to_picked[:] = distance[:, picked_sequence].sum(axis=1)

    while len(picked) < batch_size and len(candidates) > 0:
        all_pairs = max(1, len(picked) * (len(picked) + 1) / 2.0)

        # TODO: optimize - we make copy every iteration, this could be calculated as iterator
        candidates_scores = c * distances_to_picked[candidates] / all_pairs \
                            + (1 - c) * base_scores[candidates] / max(1, len(picked) + 1)

        assert all(candidates_scores <= 1.0)

        candidates_index = np.argmax(candidates_scores)
        new_index = candidates[candidates_index]
        picked.add(new_index)
        picked_sequence.append(new_index)
        del candidates[candidates_index]

        distances_to_picked += distance[:, new_index]

    # This stores (x_i, a_j), where x_i is from whole dataset and a_j is from picked subset
    # (a_i, a_j) and (a_j, a_i) - those are doubled
    picked_dissimilarity = distances_to_picked[picked_sequence].sum() / 2.0
    A = base_scores[picked_sequence].mean()
    B = (1.0 / max(1, len(picked) * (len(picked) - 1) / 2.0)) * picked_dissimilarity
    scores = (1 - c) * A \
             + c * B

    return picked_sequence, scores

_qgb_solver_numba = None

try:
    import numba
    from numba import autojit

    def _qgb_solver_numba(distance, base_scores,warm_start , c, batch_size):
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


class SetFunctionOptimizerGame(object):
    """
    Simple cast of set function optimization as UCT-compatible game

    In cluster case actions are cluster ids and state are selected ids, whereas
    in regular case actions and state are both true ids.

    Parameters
    ----------
    X: list of ids or list of list of ids

    batch_size: int
      How many samples do we want to collect

    scorer:
      Can be called, as scorer.score(ids), and returns objective function over ids.
      In cluster scenario, we should also have scorer.score_cluster_elements(cluster_id)

    element_picking_function: string, default: "max"
      Which element to pick from cluster? Possible values are "max", "prop" and "random".
    """

    def __init__(self, X, scorer, batch_size, rng, element_picking_function="max", use_scorer_playout=False):
        assert batch_size > 0, "C'mon!"
        assert element_picking_function in ['max', 'prop', 'random']

        self.rng = check_random_state(rng)
        self.scorer = scorer
        self.X = X
        self.use_scorer_playout = use_scorer_playout
        self.batch_size = batch_size
        self.element_picking_function = element_picking_function

        if isinstance(self.X[0], list):
            self.cluster_ids = set(range(len(self.X)))
            self.cluster_scores = [self.scorer.score_cluster(id) for id in self.cluster_ids]
        else:
            # Assume already all are unknown
            self.all_ids = set(range(len(X)))

    def get_actions(self, state):
        """ Returns list of actions """

        if isinstance(self.X[0], list):
            # Reduce action to just picking cluster
            return list(self.cluster_ids.difference(state['cluster_ids']))
        else:
            return list(self.all_ids.difference(state["ids"]))

    def transition(self, state, action, copy=True):
        """ Transforms state """
        if copy:
            state = {"ids": list(state['ids']), "cluster_ids": list(state['cluster_ids'])}

        if isinstance(self.X[0], list):
            state['cluster_ids'].append(action)
            state['ids'].append(self._cluster_to_element(action))
        else:
            state["ids"].append(action)

        return state

    def utility(self, state):
        """ State -> quasi greedy cost. We want to minimize it. """
        return -self.scorer.score(state['ids'])


    def playout(self, state):
        state = deepcopy(state)
        if hasattr(self.scorer, "playout") and self.use_scorer_playout:
            if isinstance(self.X[0], list):
                added_cluster_ids, added_ids = self.scorer.playout(state['ids'], state['cluster_ids'], rng=self.rng)
            else:
                added_ids = self.scorer.playout(state['ids'], rng=self.rng)
                added_cluster_ids = []
        else:
            if isinstance(self.X[0], list):
                left_ids = list(self.cluster_ids.difference(state['cluster_ids']))
                added_cluster_ids = self.rng.choice(left_ids, self.batch_size - len(state['cluster_ids']), replace=False).tolist()
                added_ids = [self._cluster_to_element(cluster_id) for cluster_id in added_cluster_ids]
            else:
                left_ids = list(self.all_ids.difference(state['ids']))
                added_ids = self.rng.choice(left_ids, self.batch_size - len(state['ids']), replace=False).tolist()
                added_cluster_ids = []

        state['cluster_ids'] += added_cluster_ids
        state['ids'] += added_ids
        if len(state['ids']) != self.batch_size:
            import pdb
            pdb.set_trace()
        assert len(state['ids']) == self.batch_size, str(state['ids']) + " " + str(added_ids)
        return state


    def playout_and_score(self, state):
        if hasattr(self.scorer, "playout_and_score") and self.use_scorer_playout:
            if isinstance(self.X[0], list):
                return -self.scorer.playout_and_score(state['ids'], cluster_ids=state['cluster_ids'], rng=self.rng)
            else:
                return -self.scorer.playout_and_score(state['ids'], rng=self.rng)
        else:
            if isinstance(self.X[0], list):
                left_ids = list(self.cluster_ids.difference(state['cluster_ids']))
                added_cluster_ids = self.rng.choice(left_ids, self.batch_size - len(state['cluster_ids']), replace=False).tolist()
                added_ids = [self._cluster_to_element(cluster_id) for cluster_id in added_cluster_ids]
                assert len(added_cluster_ids) + len(state['ids']) == self.batch_size
            else:
                left_ids = list(self.all_ids.difference(state['ids']))
                added_ids = self.rng.choice(left_ids, self.batch_size - len(state['ids']), replace=False).tolist()

        return -self.scorer.score(state['ids'] + added_ids)

    def get_key(self, state):
        # TODO: change to tuple
        return str(sorted(state['ids']))

    def is_terminal(self, state):
        return len(state["ids"]) == self.batch_size

    def _cluster_to_element(self, cluster_id):
        element_scores = self.scorer.score_cluster(cluster_id)
        assert len(element_scores) == len(self.X[cluster_id])
        if self.element_picking_function == "max":
            return self.X[cluster_id][np.argmax(element_scores)]
        elif self.element_picking_function == "prop":
            return self.rng.choice(self.X[cluster_id], p=element_scores/element_scores.sum())
        elif self.element_picking_function == "random":
            return self.rng.choice(len(self.X[cluster_id]))
        else:
            raise NotImplementedError()


class QuasiGreedyBatchScorer(object):
    """
    Implementation of scorer (to be used in SetFunctionOptimizerGame)

    ----------
    clustering: np.array, shape: (n_samples, )
      Each sample is assigned cluster id
    """

    def __init__(self, X, y, distance_cache, model, base_strategy, batch_size, c, rng, optim=2, unknown_clustering=None,
                 playout_sample_size=1, playout_repetitions=3):
        self.model = model
        self.playout_repetitions = playout_repetitions
        self.playout_sample_size = playout_sample_size
        self.optim = optim
        self.y = y
        self.batch_size = batch_size
        self.c = c
        self.base_strategy = base_strategy
        self.rng = rng
        self.distance_cache = distance_cache
        self.unknown_distance_cache = distance_cache[masked_indices(y), :][:, masked_indices(y)]
        _, base_scores_masked = base_strategy(X, self.y, rng=np.random.RandomState(self.rng),
                                              model=model, batch_size=batch_size, return_score=True)

        self.base_scores = np.zeros_like(y.data).astype("float32")
        self.base_scores[masked_indices(y)] = base_scores_masked
        self.base_scores_unknown = base_scores_masked
        self.playout_prior = base_scores_masked / base_scores_masked.sum()

        # TODO: this check is most likely not needed
        if isinstance(self.base_scores, np.ma.MaskedArray):
            self.base_scores = self.base_scores.data
            logger.warning("Strategy returned base_scores as MaskedArray")

        self.base_scores = self.base_scores.astype("float32")
        self.distance_cache = self.distance_cache.astype("float32")

        # This is slightly hacky, but ensures correct indexing
        self.unknown_clustering = unknown_clustering
        if unknown_clustering is not None:
            self._cluster_to_scores = {id: base_scores_masked[np.where(unknown_clustering == id)[0]] for id in
                                       set(list(unknown_clustering))}

    def score_cluster(self, id):
        assert self.unknown_clustering is not None, "Requires clustering parameter"
        return self._cluster_to_scores[id]

    def playout(self, ids, rng, cluster_ids=None):
        playouts = self._playout(ids, rng, cluster_ids, sample_size=0)
        scores = [p[1] for p in playouts]
        best_playout = playouts[np.argmax(scores)]
        if len(best_playout[0]) != self.batch_size:
            import pdb
            pdb.set_trace()
        assert len(best_playout[0]) == self.batch_size, "Playout has failed"
        return list(playouts[np.argmax(scores)][0][len(ids):]) # Just return added

    def playout_and_score(self, ids, rng, cluster_ids=None):
        """
        Score ids by running QGB from a sum of ids and a sample_size sample k times
        Sampling is performed proportionally to uncertainty
        """
        playouts = self._playout(ids, rng, cluster_ids, sample_size=self.playout_sample_size)
        scores = [p[1] for p in playouts]
        return sum(scores)/len(scores)

    def score(self, ids, remap=True):
        if remap:
            # Ids are in "unknown" indexing if remap is passed
            ids = masked_indices(self.y)[ids]

        assert len(ids) >= 2, "little ids" + str(ids)  # Otherwise there are no pairs

        if self.optim == 1:
            all_pairs_x, all_pairs_y = zip(*list(combinations(ids, r=2)))# zip(*product(ids, ids))
            # Product has n^2 while correct number is n * (n - 1) / 2.0
            # all_pairs = (len(ids) * (len(ids) - 1))
            # Should have this length
            all_pairs = (len(ids) * (len(ids) - 1)) / 2.0
            assert len(all_pairs_x) == all_pairs
            return (1. - self.c) * self.base_scores[ids].mean() + \
                   (self.c / all_pairs) * self.distance_cache[all_pairs_x, all_pairs_y].sum()
        elif self.optim == 2:
            return score_cython(np.array(ids).astype("int"), self.distance_cache, self.base_scores, self.c)
        else:
            all_pairs_x, all_pairs_y =  zip(*product(ids, ids))
            # Product has n^2 while correct number is n * (n - 1) / 2.0
            all_pairs = (len(ids) * (len(ids) - 1))
            return (1. - self.c) * self.base_scores[ids].mean() + \
                   (self.c / all_pairs) * self.distance_cache[all_pairs_x, all_pairs_y].sum()

    def _score_individual(self, ids, remap=True):
        """
        Returns array of UNC and DIST scores for each sample
        """
        if remap:
            # Ids are in "unknown" indexing if remap is passed
            ids = masked_indices(self.y)[ids]

        all_pairs_x, all_pairs_y =  zip(*product(ids, ids))
        # Product has n^2 while correct number is n * (n - 1) / 2.0
        all_pairs = (len(ids) * (len(ids) - 1))
        uncert_scores = self.base_scores[ids]
        dist_scores = (1. / all_pairs) * np.ones_like(uncert_scores)
        for id, i in enumerate(ids):
            dist_scores[id] *= self.distance_cache[i, ids].sum()
        return uncert_scores, dist_scores

    def _playout(self, ids, rng, cluster_ids, sample_size):
        """
        Score ids by running QGB from a sum of ids and a sample_size sample k times
        Sampling is performed proportionally to uncertainty
        """

        assert not cluster_ids, "Not implemented"
        assert self.batch_size - len(ids) >= 0

        if self.batch_size - len(ids) == 0:
            return [list(ids) for _ in range(self.playout_repetitions)]

        playouts = []

        if self.optim <= 1:
            f = _qgb_solver_python
        elif self.optim == 2:
            f = _qgb_solver_numba
        else:
            raise NotImplementedError("We cannot be THAT optimized..")

        # Can't sample too much (or put it differently, z pustego i Salomon nie naleje)
        sample_size = self.playout_sample_size
        sample_size = min(self.base_scores_unknown.shape[0] - len(ids), sample_size)
        # This is tricky: we don't want to sample last id
        sample_size = min(self.batch_size - (1 + len(ids)), sample_size)

        #
        # # mask some of the probability mass in an efficient manner
        # masked_p = self.playout_prior[ids]
        # self.playout_prior[ids] = 0
        # self.playout_prior /= self.playout_prior.sum()

        if sample_size:
            ids_set = set(ids)
            candidates = [i for i in xrange(self.playout_prior.shape[0]) if i not in ids_set]
            # TODO: optimize
            playout_prior = self.playout_prior[candidates]
            playout_prior /= playout_prior.sum()

        for _ in range(self.playout_repetitions):
            ids_sample = list(ids)
            if sample_size > 0:
                ids_sample += list(rng.choice(candidates, sample_size, p=playout_prior, replace=False))

            if self.optim == 2:
                ids_sample = np.array(ids_sample, dtype=np.int32)

            playout = f(self.unknown_distance_cache, self.base_scores_unknown, ids_sample, self.c, self.batch_size)
            playouts.append(playout)
            if len(playout[0]) != self.batch_size:
                import pdb
                pdb.set_trace()
            assert len(playout[0]) == self.batch_size

        # # unmask playout_prior
        # self.playout_prior *= (1 + masked_p.sum())
        # self.playout_prior[ids] = masked_p

        return playouts