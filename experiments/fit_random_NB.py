from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np

protein = '5ht6'
fingerprint = "ExtFP"
seed = 666

strategies = [('random_query', {}),
              ('uncertainty_sampling', {}),
              ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))})]

nb_uncertainty = run_experiment("fit_grid",
                                recalculate_experiments=True,
                                n_jobs = 8,
                                experiment_detailed_name="fit_NB_uncertainty_%s_%s" % (protein, fingerprint),
                                base_experiment="fit_active_learning",
                                seed=666,
                                base_experiment_kwargs={"strategy": "uncertainty_sampling",
                                                        "loader_function": "get_splitted_data",
                                                        "batch_size": 20,
                                                        "base_model": "RandomNB",
                                                        "loader_args": {"n_folds": 2,
                                                                        "seed": seed},
                                                        "param_grid": {'h': list(np.linspace(100,500,5))}})