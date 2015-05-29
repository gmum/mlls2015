import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(__file__))
import misc
from misc.config import *
import kaggle_ninja
from kaggle_ninja import *
from collections import namedtuple
import random_query

from utils import *
import copy
import sklearn

def run_experiment_grid(name, grid_params, recalculate=False, timeout=-1, n_jobs=2,  **kwargs):
    """
    :param name: passed to run_experiment, name of experiment
    :param grid_params: ex. {"C": [10,20,30,40]}, note - might have dot notation dataset.path.X = [..]
    :param kwargs: passed to run_experiment
    :return: list of results. Result is a dict, might be almost empty for timedout results
    """
    def gen_params():
        param_list = list(sklearn.grid_search.ParameterGrid(grid_params))
        for i, param in enumerate(param_list):
            yield param

    params = list(gen_params())

    main_logger.info("Fitting "+name+" for "+str(len(params))+" parameters combinations")

    pool = Pool(n_jobs)
    tasks = []
    for i, params in enumerate(params):
        call_params = copy.deepcopy(kwargs)
        call_params.update(params)
        call_params["name"] = name
        call_params['timeout'] = timeout
        call_params['force_reload'] = recalculate
        # Abortable is called mainly for problem with pickling functions in multiprocessing. Not important
        # timeout is passed as -1 anyway.
        tasks.append(pool.apply_async(partial(abortable_worker, "run_experiment_by_name", func_kwargs=call_params,\
                                              worker_id=i, timeout=-1)))
    pool.close()

    def progress(tasks):
        return np.mean([task.ready() for task in tasks])

    def pull_results(tasks):
        return [task.get(0) for task in tasks if task.ready()]

    while progress(tasks) != 1.0:
        try:
            for t in tasks:
                if not t.ready():
                    t.get(10) # First to fail will throw TiemoutErro()
        except TimeoutError:
            main_logger.info(str(progress(tasks)*100) + "% done")
            sys.stdout.flush()
            sys.stderr.flush()
        except KeyboardInterrupt:
            main_logger.info(name+": interrupting, killing jobs")
            sys.stdout.flush()
            sys.stderr.flush()
            pool.terminate()
            pool.join()
            raise ValueError("raising value to prevent caching")
    pool.terminate()
    pool.join()
    # Cache results with timeout
    results = pull_results(tasks)
    return results

def run_experiment_by_name(name, **kwargs):
    # Note: this line might cause some problems with path. Add experiments folder to your path
    return run_experiment(__import__(name).ex, **kwargs)

kaggle_ninja.register("run_experiment_by_name", run_experiment_by_name)



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
