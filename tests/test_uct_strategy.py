# -*- coding: utf-8 -*-
"""
 Simple test for UCT correctness
"""
import sys
sys.path.append("..")
from sklearn.linear_model import Perceptron
from alpy.utils import unmasked_indices, masked_indices
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron
from alpy.utils import mask_unknowns, unmasked_indices, masked_indices
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances
def normalized_euclidean_pairwise_distances(X):
    """ d(i, j) = ||xi - xj||/max(||xk - xl||)"""
    D = pairwise_distances(X, metric="euclidean")
    return D / D.max()

def exponential_pairwise_distances(X):
    """ d(i, j) = exp(-||xi - xj||)"""
    D = pairwise_distances(X, metric="euclidean")
    return 1 - np.exp(-D)
from models.uct import *

def test_cross_and_circle():
    def playout_uct_vs_random(budget=100, policy=ucb_policy):
        game = CrossAndCircle()

        state = {"board": np.array([["x", "x", "x"], ["x", "x", "x"], ["x", "x", "x"]], dtype=object),
                 "player": 0}

        while not game.is_terminal(state):
            if state['player'] == 0:
                uct = UCT(game=game, N=budget, policy=policy)
                uct.fit(state)
                state = game.transition(state, uct.best_action_)
            else:
                state = game.transition(state, random.choice(game.get_actions(state)))

        return game.utility(state)[0] == 1

    budget = 60
    wins_0 = 0
    for i in range(30):
        wins_0 += playout_uct_vs_random(budget=budget)

    assert wins_0 >= 20, "Random player should be consistently worse"

def test_uct_strategy_optimizer():

    ### Generate necessary data (TODO: should go into env later) ###
    def gauss_sample(N, K, D=2):
        mean = np.random.uniform(-10,10, size=(N,D))
        dev = np.random.uniform(0.1, 2, size=(N,))
        size = np.random.uniform(0.5*K, 1.5*K, size=(N,))
        X = []
        for m,d,s in zip(mean, dev,size):
            X.append(np.random.normal(m, d, size=(K,D)))
        return np.vstack(X)

    X = gauss_sample(5, 100, 2)
    hyperplane = np.random.uniform(0,1,size=(3,))
    y = np.array([hyperplane[0:2].dot(p.reshape(-1,1)) + hyperplane[2] >= 0 for p in X]).reshape(-1)
    samples = np.random.choice(range(X.shape[0]), 100)
    y = mask_unknowns(y, range(len(y)))
    y[samples] = y.data[samples] # Unmask part of data

    distance_cache = normalized_euclidean_pairwise_distances(X)
    model = Perceptron().fit(X[unmasked_indices(y)], y[unmasked_indices(y)])
    batch_size = 20
    c = 0.3
    rng = 777
    base_strategy = UncertaintySampling()
    scorer = QuasiGreedyBatchScorer(X, y, distance_cache=distance_cache, base_strategy=base_strategy,
                                    model=model, c=c, rng=rng, batch_size=batch_size)
    game = UCTStrategyOptimizer(rng=777, X=X, y=y, scorer=scorer, batch_size=batch_size)

    assert game.is_terminal({"ids": range(20)}) == True, "Discovers correctly terminal state"
    assert game.is_terminal({"ids": range(18)}) == False, "Discovers correctly terminal state"

    state = {"ids": masked_indices(y)[0:-2]}
    state['cluster_ids'] = list(state['ids'])
    actions = game.get_actions(state)
    assert masked_indices(y)[-1] in actions and masked_indices(y)[-2] in actions, "Select correctly unknown ids"

    state = {"ids": list(masked_indices(y)[0:-2])}
    state['cluster_ids'] = list(state['ids'])
    state = game.transition(state, masked_indices(y)[-1], copy=True)
    assert masked_indices(y)[-1] in state['ids'] and masked_indices(y)[-2] not in state['ids'], "Correct transition"

    state = {"ids": list(masked_indices(y)[0:1])}
    state['cluster_ids'] = list(state['ids'])
    terminal_state = game.playout(state)
    assert len(terminal_state['ids']) == 20, "Played out state to the end"
    assert set(terminal_state['ids']) <= set(masked_indices(y))

    # Check that utility is consistent with QB scoring
    strategy = QuasiGreedyBatch(distance_cache=distance_cache, c=c)
    for i in range(10):
        strategy_ids, score = strategy(X, y, rng=np.random.RandomState(i), sample_first=True,
                                       model=model, batch_size=batch_size,
                                       return_score=True)
        state = {"ids": strategy_ids}
        score_game = -game.utility(state)
        assert np.abs(score - score_game) < 1e-2, "QGB score is consistent"

    strategy = QuasiGreedyBatch(distance_cache=distance_cache, n_tries=10, c=0.3)
    strategy_ids, qgb_score = strategy(X, y, rng=np.random.RandomState(777),
                                   model=model, batch_size=20, return_score=True)
    state = {"ids": []}
    best_so_far = [0]
    for i in range(20):
        terminal_state = game.playout(state)
        score = -game.utility(terminal_state)
        best_so_far.append(max(best_so_far[-1], score))
    assert best_so_far[-1] <= qgb_score, "Random sampling is much worse than QGB"