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

    strategies = [('random_query', {}),
                  ('uncertainty_sampling', {}),
                  ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))}),
                  ('chen_krause', {})
                  ]

    preprocess_fncs = [["to_binary", {"all_below": True}]]

    for strat, strat_grid in strategies:
        svmtan_exp = run_experiment("fit_grid",
                                    recalculate_experiments=True,
                                    n_jobs=1,
                                    experiment_detailed_name="fit_SVMTAN_%s_%s_%s" % (strat, protein, fingerprint),
                                    base_experiment="fit_active_learning",
                                    seed=seed,
                                    grid_params=strat_grid,
                                    base_experiment_kwargs={"strategy": strat,
                                                            "preprocess_fncs": preprocess_fncs,
                                                            "protein": protein,
                                                            "fingerprint": fingerprint,
                                                            "warm_start_percentage": warm_start_percentage,
                                                            "batch_size": batch_size,
                                                            "base_model": "SVMTAN",
                                                            "loader_function": loader[0],
                                                            "loader_args": loader[1],
                                                            "param_grid": param_grid})

if __name__ == '__main__':

    assert len(sys.argv) == 3, "pass one protein and batch_size"
    protein = sys.argv[1]
    batch_size = int(sys.argv[2])
    assert protein in proteins, "please pick one of proteins: %s" % proteins
    assert batch_size in [10, 20, 100]

    run(protein, batch_size)
