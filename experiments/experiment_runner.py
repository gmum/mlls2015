import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import misc
from misc.config import *
import kaggle_ninja
from kaggle_ninja import *
from collections import namedtuple

from utils import *
import copy
import sklearn

import models
from itertools import chain
from models.utils import ObstructedY
from collections import defaultdict
from sklearn.metrics import auc
import socket
import json

def fit_AL_on_folds(model_cls, folds):
    metrics = defaultdict(list)
    monitors = []
    mean_train_metric = []
    for i in range(len(folds)):
        model = model_cls()

        X = folds[i]['X_train']
        y = folds[i]['Y_train']["data"]
        y_obst = ObstructedY(y)

        X_valid = folds[i]['X_valid']
        y_valid = folds[i]['Y_valid']["data"]

        test_error_datasets = [("concept", (X_valid["data"], y_valid))]
        model.fit(X, y_obst, test_error_datasets=test_error_datasets)
        y_valid_pred = model.predict(X_valid["data"])
        y_pred = model.predict(X["data"])

        for metric_name, metric_value in chain(
                binary_metrics(y_valid, y_valid_pred, "valid").items(),
                binary_metrics(y, y_pred, "train").items()):

            metrics[metric_name].append(metric_value)

        monitors.append(copy.deepcopy(model.monitors))

        train_metric = model.metrics[0].__name__ + '_concept'
        mean_train_metric.append(model.monitors[train_metric])

    keys = metrics.keys()
    for k in keys:
        metrics["mean_"+k] = np.mean(metrics[k])

    mean_train_metric = np.array(mean_train_metric).mean(axis=0)
    metrics['auc'] = auc(np.arange(mean_train_metric.shape[0]), mean_train_metric)

    return metrics, monitors

def run_experiment_grid(name, grid_params, logger=main_logger, timeout=-1, n_jobs=2,  **kwargs):
    """
    :param name: passed to run_experiment, name of experiment
    :param grid_params: ex. {"C": [10,20,30,40]}, note - might have dot notation dataset.path.X = [..]
    :param kwargs: passed to run_experiment
    :return: list of results. Result is a dict, might be almost empty for timedout results
    """


    def gen_params():
        # This is hack that enablesus to use ParameterGrid
        param_list = list(sklearn.grid_search.ParameterGrid(grid_params))
        for i, param in enumerate(param_list):
            yield  {k.replace(":", "."): v for (k,v) in param.items()}

    params = list(gen_params())
    import sys
    sys.stdout.flush()
    logger.info("Fitting "+name+" for "+str(len(params))+" parameters combinations")

    pool = Pool(n_jobs)
    tasks = []
    for i, params in enumerate(params):
        call_params = copy.deepcopy(kwargs)
        call_params.update(params)
        call_params["name"] = name
        call_params["experiment_detailed_name"] = kwargs.get("experiment_detailed_name")+"_subfit"
        call_params['timeout'] = timeout
        # Abortable is called mainly for problem with pickling functions in multiprocessing. Not important
        # timeout is passed as -1 anyway.
        tasks.append(pool.apply_async(partial(abortable_worker, "run_experiment", func_kwargs=call_params,\
                                              worker_id=i, timeout=-1)))
    pool.close()

    def progress(tasks):
        return np.mean([float(task.ready()) for task in tasks])

    def pull_results(tasks):
        return [task.get(0) for task in tasks if task.ready()]


    last_dump = 0
    start_time = time.time()
    info_file = os.path.join(c["BASE_DIR"],kwargs["experiment_detailed_name"] + ".info")
    partial_results_file = os.path.join(c["BASE_DIR"],kwargs["experiment_detailed_name"] + "_partial.pkl")
    os.system("rm " + info_file)
    os.system("rm " + partial_results_file)

    def dump_results(start_time, last_dump):
        current_progress = progress(tasks)

        elapsed = time.time() - start_time

        call="run_experiment_grid(name={0}, grid_params={1}, ".format(name, grid_params) + \
                ",".join([str(k)+"="+str(v) for k,v in kwargs.items()]) + ")"

        partial_result_info = {"progress": progress(tasks),
                               "heartbeat": time.time(),
                               "call": call,
                               "hostname": socket.gethostname(),
                          "name": kwargs["experiment_detailed_name"],
                          "elapsed_time":elapsed,
                          "projected_time":elapsed/current_progress}

        with open(info_file, "w") as f:
            f.write(json.dumps(partial_result_info))

        if current_progress - last_dump > 0.1:
            partial_results = pull_results(tasks)
            pickle.dump(partial_results, open(partial_results_file,"w"))
            return current_progress

        return last_dump

    while progress(tasks) != 1.0:
        try:
            for t in tasks:
                if not t.ready():
                    t.get(10) # First to fail will throw TiemoutErro()
        except TimeoutError:
            logger.info(str(progress(tasks)*100) + "% done")
            last_dump = dump_results(start_time, last_dump)
            sys.stdout.flush()
            sys.stderr.flush()
        except KeyboardInterrupt:
            logger.info(name+": interrupting, killing jobs")
            sys.stdout.flush()
            sys.stderr.flush()
            os.system("rm " + info_file)
            os.system("rm " + partial_results_file)
            pool.terminate()
            pool.join()
            raise ValueError("raising value to prevent caching")

    dump_results(start_time, last_dump)
    pool.terminate()
    pool.join()
    # Cache results with timeout
    results = pull_results(tasks)
    return results

def run_experiment(name, **kwargs):
    ex = find_obj(name)
    ex.logger = get_logger(ex.name)
    return ex.run(config_updates=kwargs).result

kaggle_ninja.register("run_experiment", run_experiment)



# ## Misc ##
#
# is_primitive = lambda v: isinstance(v, (int, float, bool, str))
#
# def _replace_in_json(x, sub_key, sub_value, prefix=""):
#     """
#     Replaces key-value pair in complex structure using dot notation (composed on list, dict and primitives)
#     @returns replaced_structure, how_many_replaced (int)
#     >>>    grid = {"C": [10,20,30], "dataset": {"path": 20}, "values": [1,2]}
#     >>>    print _replace_in_json(grid, "dataset.path", 10)
#     >>>    print _replace_in_json(grid, "C", 10)
#     """
#
#     # Replaces keys in complex structure of list/values/dicts
#     if x is None:
#         return x, 0
#
#     if is_primitive(x):
#         if sub_key + "." == prefix:
#             return sub_value, 1
#         else:
#             return x, 0
#     elif isinstance(x, dict):
#         if prefix + sub_key.split(".")[-1] == sub_key:
#             x[sub_key.split(".")[-1]] = sub_value
#             return x, 1
# #         for key in x:
# #             if prefix+key == sub_key:
# #                 x[key] = sub_value
# #                 return x, 1
#         replaced = dict({k: _replace_in_json(v, sub_key, sub_value, prefix + k + ".") for k,v in x.iteritems()})
#         return dict({k: v[0] for k,v in replaced.iteritems()}), sum(v[1] for v in replaced.itervalues())
#     elif isinstance(x, list):
#         replaced = [_replace_in_json(v, sub_key, sub_value, prefix=prefix + str(id) + ".") for id, v in enumerate(x)]
#         return [r[0] for r in replaced], sum(r[1] for r in replaced)
#     else:
#         raise NotImplementedError("Not supported argument type")
