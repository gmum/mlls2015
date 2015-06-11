from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np
import sys
from get_data import proteins

def run(protein, batch_size):

    fingerprint = "ExtFP"
    seed = 666
    warm_start_percentage = 0.05
    param_grid = {'C': list(np.logspace(-3, 4, 8))}

    loader = ["get_splitted_data_clusterwise", {
        "seed": seed,
        "valid_size": 0.1,
        "n_folds": 5}]

    strategies = [('czarnecki', {"strategy_projection_h": [0]}),
                  ('quasi_greedy_batch', {"C": [0.5]})
                  ]

    preprocess_fncs = [["to_binary", {"all_below": True}]]


    for strat, strat_grid in strategies:
        for value in strat_grid[strat_grid.keys()[0]]:
            kwargs = {"strategy": strat,
                                                                "preprocess_fncs": preprocess_fncs,
                                                                "protein": protein,
                                                                "fingerprint": fingerprint,
                                                                "warm_start_percentage": warm_start_percentage,
                                                                "batch_size": batch_size,
                                                                "base_model": "SVMTAN",
                                                                "loader_function": loader[0],
                                                                "loader_args": loader[1],
                                                                "param_grid": param_grid}
            kwargs.update({strat_grid.keys()[0]: value})
            svmtan_exp = run_experiment("fit_grid",
                                        recalculate_experiments=False,
                                        n_jobs=4,
                                        experiment_detailed_name="fit_SVMTAN_%s_%s_%s_%s_%s=%s" % \
                                                                 (strat, protein, fingerprint,
                                                                  str(batch_size),
                                                                  str(strat_grid.keys()[0]),
                                                                  str(value) \
                                            ),
                                        base_experiment="fit_active_learning",
                                        seed=seed,
                                        grid_params=strat_grid,
                                        base_experiment_kwargs=kwargs)

run("cathepsin", 50)