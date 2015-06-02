from experiments.experiment_runner import run_experiment, run_experiment_grid
from experiments import experiment_runner, fit_active_learning, fit_grid
from sklearn.svm import SVC

from experiments.utils import plot_grid_experiment_results, get_best, plot_monitors

from kaggle_ninja import *
turn_on_force_reload_all()

grid_result_passive = run_experiment("fit_grid",
                                    recalculate_experiments=True,
                                    n_jobs = 4, 
                                    experiment_detailed_name="fit_svm_passive_tiles",
                                    base_experiment="fit_active_learning",
                                    seed=666,
                                    grid_params = {"base_model_kwargs:C": list(np.logspace(-5,5,10)),
                                                   "base_model_kwargs:kernel": ['linear']},
                                    base_experiment_kwargs={"strategy": "random_query",
                                                       "loader_function": "get_splitted_uniform_data",
                                                       "batch_size": 20, \
                                                       "base_model": "SVC",
                                                       "loader_args": {"n_folds": 2}})

passive_exp = get_best(grid_result_passive.experiments, "mean_mcc_valid")