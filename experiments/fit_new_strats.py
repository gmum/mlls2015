from experiment_runner import run_experiment
from experiments import fit_grid
import numpy as np
import sys
from get_data import proteins

def run(protein, batch_size, fingerprint="PubchemFP"):

    seed = 666
    warm_start_percentage = 0.05
    param_grid = {'C': list(np.logspace(-3, 4, 8))}

    loader = ["get_splitted_data_clusterwise", {
        "seed": seed,
        "valid_size": 0.1,
        "n_folds": 5}]


    preprocess_fncs = [["to_binary", {"all_below": True}]]


    run_experiment("fit_grid",
                   force_reload=True,
                   recalculate_experiments=False,
                   n_jobs=5,
                   experiment_detailed_name="fit_SVMTAN_random_query_%s_%s_%s" % (protein, fingerprint, str(batch_size)),
                   base_experiment="fit_active_learning",
                   seed=seed,
                   grid_params={},
                   base_experiment_kwargs={"strategy": "random_query",
                                           "preprocess_fncs": preprocess_fncs,
                                           "protein": protein,
                                           "fingerprint": fingerprint,
                                           "warm_start_percentage": warm_start_percentage,
                                           "batch_size": batch_size,
                                           "base_model": "SVMTAN",
                                           "loader_function": loader[0],
                                           "loader_args": loader[1],
                                           "param_grid": param_grid})

    run_experiment("fit_grid",
                   force_reload=True,
                   recalculate_experiments=False,
                   n_jobs=5,
                   experiment_detailed_name="fit_SVMTAN_uncertainty_sampling_%s_%s_%s" % (protein, fingerprint, str(batch_size)),
                   base_experiment="fit_active_learning",
                   seed=seed,
                   grid_params={},
                   base_experiment_kwargs={"strategy": "uncertainty_sampling",
                                           "preprocess_fncs": preprocess_fncs,
                                           "protein": protein,
                                           "fingerprint": fingerprint,
                                           "warm_start_percentage": warm_start_percentage,
                                           "batch_size": batch_size,
                                           "base_model": "SVMTAN",
                                           "loader_function": loader[0],
                                           "loader_args": loader[1],
                                           "param_grid": param_grid})

    for c in list(np.linspace(0.1, 0.9, 9)):
        svmtan_exp = run_experiment("fit_grid",
                                force_reload=False,
                                recalculate_experiments=False,
                                n_jobs=5,
                                experiment_detailed_name="fit_SVMTAN_czarnecki_two_clusters_c_%.2f_%s_%s_%s" % (c, protein, fingerprint, str(batch_size)),
                                base_experiment="fit_active_learning",
                                seed=seed,
                                grid_params={"strategy_kwargs:c": [c]},
                                base_experiment_kwargs={"strategy": "czarnecki_two_clusters",
                                                        "preprocess_fncs": preprocess_fncs,
                                                        "protein": protein,
                                                        "fingerprint": fingerprint,
                                                        "warm_start_percentage": warm_start_percentage,
                                                        "batch_size": batch_size,
                                                        "base_model": "SVMTAN",
                                                        "loader_function": loader[0],
                                                        "loader_args": loader[1],
                                                        "param_grid": param_grid})

    run_experiment("fit_grid",
                   force_reload=True,
                   recalculate_experiments=False,
                   n_jobs=5,
                   experiment_detailed_name="fit_SVMTAN_czarnecki_%s_%s_%s" % (protein, fingerprint, str(batch_size)),
                   base_experiment="fit_active_learning",
                   seed=seed,
                   grid_params={},
                   base_experiment_kwargs={"strategy": "czarnecki",
                                           "preprocess_fncs": preprocess_fncs,
                                           "protein": protein,
                                           "fingerprint": fingerprint,
                                           "warm_start_percentage": warm_start_percentage,
                                           "batch_size": batch_size,
                                           "base_model": "SVMTAN",
                                           "loader_function": loader[0],
                                           "loader_args": loader[1],
                                           "param_grid": param_grid})

    for c in list(np.linspace(0.1, 0.9, 9)):
        run_experiment("fit_grid",
                       force_reload=True,
                       recalculate_experiments=False,
                       n_jobs=5,
                       experiment_detailed_name="fit_SVMTAN_multiple_pick_best_prim_c_%.2f_%s_%s_%s" % (c, protein, fingerprint, str(batch_size)),
                       base_experiment="fit_active_learning",
                       seed=seed,
                       grid_params={"strategy_kwargs:c": [c]},
                       base_experiment_kwargs={"strategy": 'multiple_pick_best',
                                               "preprocess_fncs": preprocess_fncs,
                                               "protein": protein,
                                               "fingerprint": fingerprint,
                                               "warm_start_percentage": warm_start_percentage,
                                               "batch_size": batch_size,
                                               "base_model": "SVMTAN",
                                               "loader_function": loader[0],
                                               "loader_args": loader[1],
                                               "param_grid": param_grid})


    for c in list(np.linspace(0.1, 0.9, 9)):
        run_experiment("fit_grid",
                       force_reload=True,
                       recalculate_experiments=False,
                       n_jobs=5,
                       experiment_detailed_name="fit_SVMTAN_quasi_greedy_batch_c_%.2f_%s_%s_%s" % (c, protein, fingerprint, str(batch_size)),
                       base_experiment="fit_active_learning",
                       seed=seed,
                       grid_params={"strategy_kwargs:c": [c]},
                       base_experiment_kwargs={"strategy": 'quasi_greedy_batch',
                                               "preprocess_fncs": preprocess_fncs,
                                               "protein": protein,
                                               "fingerprint": fingerprint,
                                               "warm_start_percentage": warm_start_percentage,
                                               "batch_size": batch_size,
                                               "base_model": "SVMTAN",
                                               "loader_function": loader[0],
                                               "loader_args": loader[1],
                                               "param_grid": param_grid})
    for h in [50,200,500]:
        run_experiment("fit_grid",
                       force_reload=True,
                       recalculate_experiments=False,
                       n_jobs=5,
                       experiment_detailed_name="fit_SVMTAN_multiple_pick_best_h_%d_%s_%s_%s" % (h, protein, fingerprint, str(batch_size)),
                       base_experiment="fit_active_learning",
                       seed=seed,
                       grid_params={"strategy_projection_h": [h]},
                       base_experiment_kwargs={"strategy": 'chen_krause',
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

    assert len(sys.argv) >= 3, "pass one protein and batch_size"
    protein = sys.argv[1]
    batch_size = int(sys.argv[2])
    fingerprint = "PubchemFP"
    if len(sys.argv) == 4:
        fingerprint = int(sys.argv[3])
    assert protein in proteins, "please pick one of proteins: %s" % proteins
    assert batch_size in [20,50, 100]

    if fingerprint == "all":
        run(protein, batch_size, "ExtFP")
        run(protein, batch_size, "PubchemFP")
        run(protein, batch_size, "MACCSFP")
    else:
        run(protein, batch_size, fingerprint)