import sys
sys.path.append("..")
import misc
from misc.config import *
from kaggle_ninja import *
import pandas as pd
from experiments.utils import *
from collections import Counter


def load_results():

    proteins = ['5ht7','hiv_integrase','h1','cathepsin','M1', '5ht6']
    strategies = ['uncertainty_sampling', 'quasi_greedy_batch', 'random_query', 'czarnecki', 'czarnecki_two_clusters']
    batch_sizes = [20, 50, 100]
    fingerprint = "ExtFP"

    experiments = {p+'_'+str(bs): [] for p in proteins for bs in batch_sizes}

    for p in proteins:
        for batch_size in batch_sizes:
            for strat in strategies:
                    # if batch_size != 50:
                    try:
                        exp_name = "fit_SVMTAN_%s_%s_%s_%s" % (strat, p, fingerprint, str(batch_size))
                        exp = get_experiment_results(exp_name)
                        experiments[p+'_'+str(batch_size)] += exp.experiments
                    except:
                        if batch_size == 50 and strat == "quasi_greedy_batch":
                            continue # Ignoring 50 because we started using differnt aming then
                        print "Coundn't load", p, strat, batch_size
                        continue
            for c in list(np.linspace(0.1, 0.9, 9)):
                exp_name = "fit_SVMTAN_multiple_pick_best_c_%.2f_%s_%s_%s" % (c, p, fingerprint, str(batch_size))
                try:
                    exp = get_experiment_results(exp_name)
                    assert len(exp.experiments) == 1
                    experiments[p+'_'+str(batch_size)].append(exp.experiments[0])
                except:
                    print "Couldn't load", p, "multiple", batch_size, c
                    continue
                if batch_size == 50:
                    exp_name = "fit_SVMTAN_quasi_greedy_batch_c_%.2f_%s_%s_%s" % (c, p, fingerprint, str(batch_size))
                    try:
                        exp = get_experiment_results(exp_name)
                        assert len(exp.experiments) == 1
                        experiments[p+'_'+str(batch_size)].append(exp.experiments[0])
                    except:
                        print "Couldn't load", p, "quasi_greedy", batch_size, c
                        continue

            for h in [50,200,500]:
                exp_name = "fit_SVMTAN_multiple_pick_best_h_%d_%s_%s_%s" % (h, p, fingerprint, str(batch_size))
                try:
                    exp = get_experiment_results(exp_name)
                    assert len(exp.experiments) == 1
                    experiments[p+'_'+str(batch_size)].append(exp.experiments[0])
                except:
                    print "Couldn't load", p, "czarnecki", batch_size, h
                    continue

    return experiments


def count_wins(experiments, metric = 'auc_wac_score_concept'):

    strategies = ['chen_krause', 'uncertainty_sampling', 'quasi_greedy_batch', 'random_query', 'multiple_pick_best',
              'czarnecki', 'czarnecki_two_clusters']

    wins = {s: 0 for s in strategies}
    diffs = {s: 0 for s in strategies}
    best_exps = {p: None for p in experiments.keys()}

    for p, protein_exps in experiments.iteritems():
        best_score = (None, 0)
        for e in protein_exps:
            if e.results[metric] >= best_score[1]:
                best_score = (e.config['strategy'], e.results[metric])
                best_exps[p] = e

        wins[best_score[0]] += 1
        diffs[best_score[0]] += best_score[1] - max([e.results[metric] for e in protein_exps
                                                     if e.config['strategy'] != best_score[0]])

    wins_df = pd.DataFrame([(key, wins[key], diffs[key]) for key in wins.keys()])
    wins_df.columns = ['strategy', '# wins', 'diff']

    which_df = pd.DataFrame([(p, e.config['strategy']) for p, e in best_exps.iteritems()])
    which_df.columns = ["exp", "best strategy"]
    which_df.sort('exp')

    return wins_df, which_df, best_exps


def count_strat_experiments(experiments):
    strats = []

    for key, protein_experiments in experiments.iteritems():
        for e in protein_experiments:
            strats.append(e.config['strategy'])

    return Counter(strats)

def get_best_per_strategy(experiments, metric='auc_wac_score_concept'):

    strategies = ['chen_krause', 'uncertainty_sampling', 'quasi_greedy_batch', 'random_query',
                  'multiple_pick_best', 'czarnecki', 'czarnecki_two_clusters']
    best_protein_exp = {k: {s: [None, 0] for s in strategies} for k in experiments.keys()}

    for key, protein_experiments in experiments.iteritems():
        for e in protein_experiments:
            strat = e.config['strategy']
            if e.results[metric] >= best_protein_exp[key][strat][1]:
                best_protein_exp[key][strat][0] = e
                best_protein_exp[key][strat][1] = e.results[metric]

    return {k: {s: e[0] for s, e in exps.iteritems()} for k, exps in best_protein_exp.iteritems()}


def plot_monitors(experiments, keys='metrics', folds='mean', figsize=(15,15)):
    import matplotlib.pylab as plt

    assert folds in ['all', 'mean']
    assert keys in ['metrics', 'times']

    if keys == 'times':
        include = ['unlabeled_test_times',
                   'grid_times',
                   'strat_times',
                   'concept_test_times']
    elif keys == "metrics":
        include = []
        for metr in ["wac_score"]:
            for dataset in ["concept", "unlabeled", "cluster_A_concept", "cluster_B_concept",
                            "cluster_A_unlabeled", "cluster_B_unlabeled"]:
                include.append(metr+"_"+dataset)


    keys = [k for k in experiments.values()[0].monitors[0].keys() if k in include]

    f, axes = plt.subplots(len(keys), 1)
    f.set_figheight(figsize[1])
    f.set_figwidth(figsize[0])

    assert folds == 'mean'

    if folds == 'mean':
        for ax, key in zip(axes, keys):
            for strat, e in experiments.iteritems():
                if e is None:
                    continue
                pd.DataFrame({strat: e.misc['mean_monitor'][key]}).plot(title=key, ax=ax)
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    else:
        for ax, key in zip(axes, keys):
            for e in experiments:
                for mon in e.monitors:
                    pd.DataFrame({e.name: mon[key]}).plot(title=key, ax=ax)
                    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))