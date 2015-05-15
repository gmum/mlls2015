import hashlib
import inspect
import datetime
import sys
import json

# Import all helpers (numpy, scikit, etc)
from cached_helpers import *

from kaggle_ninja import ninja_globals
from kaggle_ninja import find_obj


def get_last_cached(prefix, number=10,  load_fnc=None, start=0):
    """
    Retrieves what was cached on the machine startig with 0 ordered by time
    """
    files = glob.glob(os.path.join(c["CACHE_DIR"], prefix+"*.args"))
    files_times = []
    for f in files:
        with open(f, "r") as fh:
            args = json.loads(fh.read())
            files_times.append([args.get("time", 0), args['args'], f[0:-5]])
    selected = list(reversed(sorted(files_times)))[0:20]

    results = []
    for f in selected:
        if load_fnc is None:
            with open(f[2] + ".pkl", "r") as fh:
                results.append([f[1], cPickle.load(fh)])
        else:
            results.append(load_fnc(f[2], ""))
    return results

def cached(save_fnc=None, load_fnc=None, check_fnc=None, skip_args=[], cached_ram=False,
           use_cPickle=False, logger=None, key_args=[]):
    """
    To make it work correctly please pass parameters to function as dict

    @param save_fnc, load_fnc function(key, returned_value)
    @param check_fnc function(key) returning True/False. Checks if already stored file
    """

    global ninja_globals

    if logger is None:
        logger = ninja_globals["logger"]

    if cached_ram:
        RuntimeWarning("Please make sure that cached value is read-only or you are aware of possible multithreaded\
                       problems")

    def _cached(func):
        def func_caching(*args, **dict_args):
            if len(args) > 0:
                raise Exception("For cached functions pass all args by dict_args (ensures cache resolution)")

            a = inspect.getargspec(func)
            if a.defaults:
                default_args = dict(zip(a.args[-len(a.defaults):],a.defaults))
                for default_arg in default_args:
                    if dict_args.get(default_arg, None) is None:
                        dict_args[default_arg] = default_args[default_arg]

            # Dump special arguments
            dict_args_original = dict(dict_args)
            dict_args_original.pop("force_reload", None)
            dict_args_original.pop("store", None)

            part_key, dumped_arguments = _generate_key(func.__name__, dict_args_original, skip_args)

            full_key = func.__name__
            for k in key_args:
                if dict_args[k]!="":
                    full_key = full_key + "_" + str(dict_args[k])
            full_key = full_key+"_"+part_key

            if not ninja_globals["force_reload_all"] and \
                    not dict_args.get("force_reload", False) and not dict_args.get("store", False)\
                    and cached_ram and full_key in ninja_globals["cache"]:
                if logger:
                    ninja_globals["logger"].debug(func.__name__+": Reading from RAM cache")
                return ninja_globals["cache"][full_key]

            cache_file_default = os.path.join(ninja_globals["cache_dir"], full_key + ".pkl")

            exists = os.path.exists(cache_file_default) if check_fnc is None \
                else check_fnc(full_key, ninja_globals["cache_dir"])

            def evaluate_and_write():
                if logger:
                    logger.debug(func.__name__+": Cache miss or force reload. Caching " + full_key)

                returned_value = func(*args, **dict_args_original)
                if save_fnc:
                    save_fnc(full_key, returned_value, ninja_globals["cache_dir"])
                else:
                    with open(cache_file_default, "w") as f:
                        if use_cPickle:
                            cPickle.dump(returned_value, f)
                        else:
                            pickle.dump(returned_value, f)

                    # Write arguments for later retrieval
                    # NOTE: of course better way would be to keep a dict..
                    with open(os.path.join(ninja_globals["cache_dir"], full_key+".args"), "w") as f:
                        f.write(json.dumps({"func_name":func.__name__, "time": _uct_timestamp(), "key":\
                                            full_key, "args": json.loads(dumped_arguments)}))

                return returned_value

            if exists and not ninja_globals["force_reload_all"] and\
                    not dict_args.get("force_reload", False) and not dict_args.get("store", False):
                if logger:
                    logger.debug(func.__name__+":Loading (pickled?) file")

                if load_fnc:
                    value = load_fnc(full_key, ninja_globals["cache_dir"])
                else:
                    # We do try here because we might have failed writing pickle file before
                    try:
                        with open(cache_file_default, "r") as f:
                            value = cPickle.load(f) if use_cPickle else pickle.load(f)
                    except:
                       value = evaluate_and_write()

            else:
               value = evaluate_and_write()

            if cached_ram:
                ninja_globals["cache"][full_key] = value

            return value

        if ninja_globals["cache_on"]:
            return func_caching
        else:
            return func

    return _cached


is_primitive = lambda v: isinstance(v, (int, float, bool, str))

def _validate_for_cached(x, skip_args=[], prefix=""):
    # Returns True/False if the x is primitive,list,dict, recursively

    if x is None:
        return x

    if is_primitive(x):
        return True
    elif isinstance(x, dict):
        if not all(isinstance(key, str) for key in x):
            return False
        return all(_validate_for_cached(v, skip_args=skip_args, prefix=prefix+k+".") for k,v in x.iteritems()\
                  if prefix+k not in skip_args\
                  )
    elif isinstance(x, list):
        return all(_validate_for_cached(v, skip_args=skip_args, prefix=prefix) for v in x)
    else:
        return False

def _clean_skipped_args(x, skip_args, prefix=""):
    # Removed keys in skip_args (x should be a dict). Format is key1.key2.key3 for nested dicts.

    if x is None:
        return x

    if is_primitive(x):
        return x
    elif isinstance(x, dict):
        return dict({k: _clean_skipped_args(v, skip_args, prefix=prefix+k+".") for k,v in x.iteritems() if prefix+k not in skip_args})
    elif isinstance(x, list):
        return [_clean_skipped_args(v, skip_args, prefix=prefix) for v in x]
    else:
        raise NotImplementedError("Not supported argument type")

from collections import OrderedDict

def _sort_all(x):
    # Removed keys in skip_args (x should be a dict). Format is key1.key2.key3 for nested dicts.

    if x is None:
        return x

    if is_primitive(x):
        return x
    elif isinstance(x, dict):
        return OrderedDict([(k, _sort_all(x[k])) for k in sorted(x.keys())])
    elif isinstance(x, list):
        return list(sorted([_sort_all(v) for v in x]))
    else:
        raise NotImplementedError("Not supported argument type")

def _generate_key(func_name, args, skip_args):
    """
    Assumes all args are primitive (int, float, bool, str) or dict, in which case it will execute
    itself recursively
    """

    if not _validate_for_cached(args, skip_args):
        raise NotImplementedError("_validate_for_cached failed")

    dumped_arguments = json.dumps(_sort_all(_clean_skipped_args(args, skip_args))).strip()

    return hashlib.sha256(func_name+"_"+dumped_arguments).hexdigest(), dumped_arguments

def read_all_calls():
    # Returns list with appropriate cache entries
    # NOTE: Pretty slow, but won't be called often so it is ok. Alternatively we could cache something
    # NOTE: it might be more sensible to rewrite it using sql, as there might be some problems with concurrency..

    global ninja_globals

    cache_dict_file = os.path.join(ninja_globals["cache_dir"], "cache_dict.txt")

    file_names = glob.glob(os.path.join(ninja_globals["cache_dir"], "*.args"))

    calls = []
    for file_name in file_names:
        f = open(file_name, "r")
        calls.append(json.loads(f.read()))
        f.close()

    return calls




def _uct_timestamp():
    d = datetime.datetime.utcnow()
    epoch = datetime.datetime(1970,1,1)
    t = (d - epoch).total_seconds()

