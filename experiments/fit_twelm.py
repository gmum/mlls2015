from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np

protein = '5ht6'
fingerprint = "ExtFP"
seed = 666

strategies = [('random_query', {}),
              ('uncertainty_sampling', {}),
              ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))})]

twelm_uncertain_1 = run_experiment("fit_grid",
                                 recalculate_experiments=True,
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


twelm_uncertain_2 = run_experiment("fit_grid",
                                 recalculate_experiments=True,
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


assert np.array_equal(twelm_uncertain_1.experiments[0].results.values(),
                      twelm_uncertain_2.experiments[0].results.values())

# twelm_uncertain = run_experiment("fit_grid",
#                                  recalculate_experiments=False,
#                                  n_jobs=2,
#                                  experiment_detailed_name="fit_TWELM_uncertain_%s_%s" % (protein, fingerprint),
#                                  base_experiment="fit_active_learning",
#                                  seed=666,
#                                  param_grid={"base_model_kwargs:strategy_kwargs:c":list(np.linspace(0.1, 0.9, 9)) },
#                                  base_experiment_kwargs={"strategy": "quasi_greedy_batch",
#                                                          "loader_function": "get_splitted_data",
#                                                          "batch_size": 20,
#                                                          "base_model": "TWELM",
#                                                          "loader_args": {"n_folds": 2,
#                                                                          "seed": seed},
#                                                          "param_grid": {'C': list(np.logspace(-3,4,7))}})