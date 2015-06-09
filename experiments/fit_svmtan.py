from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np

protein = '5ht6'
fingerprint = "ExtFP"
seed = 666
warm_start_percentage = 0.05
batch_size = 20
param_grid = {'C': list(np.logspace(-3, 4, 8))}

loader = ["get_splitted_data_clusterwise", {
    "seed": seed,
    "valid_size": 0.15,
    "n_folds": 4}]

strategies = [('random_query', {}),
              ('uncertainty_sampling', {}),
              ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))}),
              ('chen_krause', {"strategy_projection_h": [10,50,100,200] })
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
