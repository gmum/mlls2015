import sys
sys.path.append('..')

from experiment_runner import run_experiment, run_experiment_grid
import fit_active_learning, fit_grid
from utils import get_best
from models.strategy import cosine_distance_normalized

from kaggle_ninja import *

protein = '5ht6'
fingerprints = ["ExtFP"]
seed = 666

for protein, fingerprint in [(protein, fp) for fp in fingerprints]:

    run_experiment("fit_grid",
                   recalculate_experiments=True,
                   n_jobs = 8,
                   experiment_detailed_name="fit_svm_passive_%s_%s" % (protein, fingerprint),
                   base_experiment="fit_active_learning",
                   seed=666,
                   base_experiment_kwargs={"strategy": "random_query",
                                           "loader_function": "get_splitted_data",
                                           "batch_size": 20,
                                           "base_model": "LinearSVC",
                                           "loader_args": {"n_folds": 2,
                                                           "seed": seed},
                                           "param_grid": {'C': list(np.logspace(-3,4,7))},
                                           "base_model_kwargs": { "loss": 'hinge'}})


    # run_experiment("fit_grid",
    #                recalculate_experiments=True,
    #                n_jobs = 8,
    #                experiment_detailed_name="fit_svm_uncertain_%s_%s" % (protein, fingerprint),
    #                base_experiment="fit_active_learning",
    #                seed=666,
    #                base_experiment_kwargs={"strategy": "uncertainty_sampling",
    #                                        "loader_function": "get_splitted_data",
    #                                        "batch_size": 20,
    #                                        "base_model": "LinearSVC",
    #                                        "loader_args": {"n_folds": 2,
    #                                                        "seed": seed},
    #                                        "param_grid": {'C': list(np.logspace(-5,5,10))},
    #                                        "base_model_kwargs": { "loss": 'hinge'}})

    # grid_result_greedy = run_experiment("fit_grid",
    #                                     recalculate_experiments=True,
    #                                     n_jobs = 2,
    #                                     experiment_detailed_name="fit_svm_greedy_%s_%s" % (protein, fingerprints),
    #                                     base_experiment="fit_active_learning",
    #                                     seed=666,
    #                                     grid_params = {"base_model_kwargs:C": list(np.logspace(-5,5,10)),
    #                                                    "base_model_kwargs:loss": ['hinge'],
    #                                                    "strategy_kwargs:c": list(np.linspace(0.1, 0.9, 9))},
    #                                     base_experiment_kwargs={"strategy": "quasi_greedy_batch",
    #                                                        "loader_function": "get_splitted_data",
    #                                                        "batch_size": 20,
    #                                                        "base_model": "LinearSVC",
    #                                                        "loader_args": {"n_folds": 2,
    #                                                                        "seed": 666}})