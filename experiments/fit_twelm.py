from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np

protein = '5ht6'
fingerprint = "ExtFP"
seed = 666
#
# strategies = [('random_query', {}),
#               ('uncertainty_sampling', {}),
#               ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))})]

twelm_uncertain = run_experiment("fit_grid",
                                 n_jobs=8,
                                 experiment_detailed_name="fit_TWELM_uncertain_%s_%s" % (protein, fingerprint),
                                 base_experiment="fit_active_learning",
                                 seed=777,
                                 base_experiment_kwargs={"strategy": "uncertainty_sampling",
                                                         "loader_function": "get_splitted_data",
                                                         "batch_size": 20,
                                                         "base_model": "TWELM",
                                                         "loader_args": {"n_folds": 2,
                                                                         "seed": seed},
                                                         "param_grid": {'C': list(np.logspace(-3,4,7))}})

twelm_chen = run_experiment("fit_grid",
                                 n_jobs=8,
                                 experiment_detailed_name="fit_TWELM_uncertain_%s_%s" % (protein, fingerprint),
                                 base_experiment="fit_active_learning",
                                 seed=777,
                                 grid_params={"strategy_projection_h":[10,50,100,200] },
                                 base_experiment_kwargs={"strategy": "chen_krause",
                                                         "loader_function": "get_splitted_data",
                                                         "batch_size": 20,
                                                         "base_model": "TWELM",
                                                         "loader_args": {"n_folds": 2,
                                                                         "seed": seed},
                                                         "param_grid": {'C': list(np.logspace(-3,4,7))}})


twelm_quasi_greedy = run_experiment("fit_grid",
                                 n_jobs=2,
                                 experiment_detailed_name="fit_TWELM_quasi_greedy_%s_%s" % (protein, fingerprint),
                                 base_experiment="fit_active_learning",
                                 seed=666,
                                 grid_params={"strategy_kwargs:c":list(np.linspace(0.1, 0.9, 9)) },
                                 base_experiment_kwargs={"strategy": "quasi_greedy_batch",
                                                         "loader_function": "get_splitted_data",
                                                         "batch_size": 20,
                                                         "base_model": "TWELM",
                                                         "loader_args": {"n_folds": 2,
                                                                         "seed": seed},
                                                         "param_grid": {'C': list(np.logspace(-3,4,7))}})