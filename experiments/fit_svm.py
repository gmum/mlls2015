from experiments.experiment_runner import run_experiment, run_experiment_grid
from experiments import experiment_runner, fit_active_learning, fit_grid
from experiments.utils import get_best
from models.strategy import cosine_distance_normalized

from kaggle_ninja import *

protein = '5ht6'
fingerprints = ["ExtFP"]

for protein, fingerprint in [(protein, fp) for fp in fingerprints]:

    run_experiment("fit_grid",
                   recalculate_experiments=True,
                   n_jobs = 8,
                   experiment_detailed_name="fit_svm_passive_%s_%s" % (protein, fingerprints),
                   base_experiment="fit_active_learning",
                   seed=666,
                   grid_params = {"base_model_kwargs:C": list(np.logspace(-5,5,10)),
                                  "base_model_kwargs:kernel": ['linear']},
                   base_experiment_kwargs={"strategy": "random_query",
                                           "loader_function": "get_splitted_data",
                                           "batch_size": 20,
                                           "base_model": "SVC",
                                           "loader_args": {"n_folds": 2,
                                                           "seed": 666}})

    grid_result_uncertainty = run_experiment("fit_grid",
                                        recalculate_experiments=True,
                                        n_jobs = 8,
                                        experiment_detailed_name="fit_svm_uncertainty_%s_%s" % (protein, fingerprints),
                                        base_experiment="fit_active_learning",
                                        seed=666,
                                        grid_params = {"base_model_kwargs:C": list(np.logspace(-5,5,10)),
                                                       "base_model_kwargs:kernel": ['linear']},
                                        base_experiment_kwargs={"strategy": "uncertanity_sampling",
                                                           "loader_function": "get_splitted_data",
                                                           "batch_size": 20,
                                                           "base_model": "SVC",
                                                           "loader_args": {"n_folds": 2,
                                                                           "seed": 666}})

    grid_result_greedy = run_experiment("fit_grid",
                                        recalculate_experiments=True,
                                        n_jobs = 8,
                                        experiment_detailed_name="fit_svm_greedy_%s_%s" % (protein, fingerprints),
                                        base_experiment="fit_active_learning",
                                        seed=666,
                                        grid_params = {"base_model_kwargs:C": list(np.logspace(-5,5,10)),
                                                       "base_model_kwargs:kernel": ['linear'],
                                                       "strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))},
                                        base_experiment_kwargs={"strategy": "quasi_greedy_batch",
                                                           "loader_function": "get_splitted_data",
                                                           "batch_size": 20,
                                                           "base_model": "SVC",
                                                           "loader_args": {"n_folds": 2,
                                                                           "seed": 666}})