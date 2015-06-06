from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np

protein = '5ht6'
fingerprint = "ExtFP"
seed = 666

strategies = [('random_query', {}),
              ('uncertainty_sampling', {}),
              ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))})]

from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np

protein = '5ht6'
fingerprint = "ExtFP"
seed = 777

strategies = [('random_query', {}),
              ('uncertainty_sampling', {}),
              ('quasi_greedy_batch', {"strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))}),
                ('chen_krause', {"strategy_projection_h":[10,50,100,200] })
              ]

for strat, strat_grid in strategies:
    svmtan_uncertainty = run_experiment("fit_grid",
                                 n_jobs=4,
                                 experiment_detailed_name="fit_NB_%s_%s_%s" % (strat, protein, fingerprint),
                                 base_experiment="fit_active_learning",
                                 seed=seed,
                                 grid_params=strat_grid, \
                                 base_experiment_kwargs={"strategy": strat,
                                                         "loader_function": "get_splitted_data",
                                                         "batch_size": 20,
                                                         "base_model": "RandomNB",
                                                         "loader_args": {"n_folds": 2,
                                                                         "seed": seed},
                                                          "param_grid": {'h': list(np.linspace(100,500,5))}})

