# -*- coding: utf-8 -*-
"""
 Simple test for UCT correctness
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
