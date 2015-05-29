import re
import hashlib
import inspect
import datetime
import sys
import json

# Import all helpers (numpy, scikit, etc)
from cached_helpers import *

from kaggle_ninja import ninja_globals
from kaggle_ninja import find_obj

import re

def get_all_by_search_arg(prefix, query_search_args={}, query_args={},  load_fnc=None):
    """
    Retrieves what was cached on the machine startig with 0 ordered by time
    @param query_search_args dict of key->regex for matching search_args (see cached)
    @param query_args dict of key->lambda for matching regular args
    """
    global ninja_globals

    files = glob.glob(os.path.join(ninja_globals["cache_dir"], prefix+"*.args"))
    files_times = []

    # Standarize query_args
    for k, obj in query_args.iteritems():
        if not hasattr(obj, '__call__'):
            query_args[k] = lambda x: x==obj

    decode_errors = 0
    for f in files:
        with open(f, "r") as fh:
            json_dump = fh.read()
            try:
                args_file = json.loads(json_dump)
                args = args_file.get("args", {})
                search_args = args_file.get("search_args", {})

                if len(search_args) or len(query_search_args) == 0:
                    calculation_timestamp = args_file.get("time", 0)

                    if all(re.match(query_search_args.get(k, ""), search_args[k]) for k in search_args) \
                        and all(query_args.get(k, (lambda x: True))(args[k]) for k in args):
                            files_times.append([calculation_timestamp, args, search_args,  f[0:-5]])
            except ValueError:
                decode_errors += 1


    reading_errors = 0

    results = []
    for f in files_times:
        if os.path.exists(f[3] + ".pkl"):
            try:
                if load_fnc is None:
                    with open(f[3] + ".pkl", "r") as fh:
                        results.append({"timestamp": f[0],"arguments": f[1], "search_arguments": f[2], "cached_result":
                            cPickle.load(fh)})
                else:
                    results.append({"timestamp":f[0], "arguments":f[1], "search_arguments": f[2], \
                                    "cached_result": load_fnc(f[3], "")})
            except:
                reading_errors += 1
        else:
            os.system("rm "+ f[3] + ".args")

    ninja_globals["logger"].warning(str(decode_errors)+" decoding errors")
    ninja_globals["logger"].warning(str(reading_errors)+" reading pkl errors")


    return results


def get_last_cached(prefix, number=10,  load_fnc=None, start=0):
    """
    Retrieves what was cached on the machine startig with 0 ordered by time
    """
    global ninja_globals

    files = glob.glob(os.path.join(ninja_globals["cache_dir"], prefix+"*.args"))
    files_times = []
    for f in files:
        with open(f, "r") as fh:
            args = json.loads(fh.read())
            files_times.append([args.get("time", 0), args['args'],args['search_args'],  f[0:-5]])
    selected = list(reversed(sorted(files_times)))[0:20]

    results = []
    for f in selected:
        if load_fnc is None:
            with open(f[3] + ".pkl", "r") as fh:
                results.append([f[1], f[2], cPickle.load(fh)])
        else:
            results.append([f[1], f[2], load_fnc(f[3], "")])
    return results

from multiprocessing import Lock
gsutil_lock = Lock()



#TODO: add globals for strings.. this is lame.

def cached(save_fnc=None, load_fnc=None, check_fnc=None, search_args=[], skip_args=[], cached_ram=False,
           use_cPickle=False, logger=None, cache_google_cloud=False, key_args=[]):
    """
    To make it work correctly please pass parameters to function as dict

    @param save_fnc, load_fnc function(key, returned_value)
    @param check_fnc function(key) returning True/False. Checks if already stored file
    """

    skip_args += search_args
    global ninja_globals

    if logger is None:
        logger = ninja_globals["logger"]

    if cached_ram:
        RuntimeWarning("Please make sure that cached value is read-only or you are aware of possible multithreaded\
                       problems")

    def _cached(func):
        def func_caching(*args, **dict_args):
            force_reload = any(re.match(expr, func.__name__) is not None for expr in ninja_globals["force_reload"]) \
                or dict_args.get("force_reload", False)


            if len(args) > 0:
                raise Exception("For cached functions pass all args by dict_args (ensures cache resolution)")

            # Get arguments including defaults
            a = inspect.getargspec(func)
            if a.defaults:
                default_args = dict(zip(a.args[-len(a.defaults):],a.defaults))
                for default_arg in default_args:
                    if dict_args.get(default_arg, None) is None:
                        dict_args[default_arg] = default_args[default_arg]

            # Generate key
            dict_args_original = dict(dict_args)
            dict_args_original.pop("force_reload", None)
            dict_args_original.pop("store", None)
            dict_args_original.pop("_write_to_cache", None)
            dict_args_original.pop("_load_cache_or_fail", None)
            part_key, dumped_arguments = _generate_key(func.__name__, dict_args_original, skip_args)
            full_key = func.__name__
            for k in key_args:
                if dict_args[k]!="":
                    full_key = full_key + "_" + str(dict_args[k])
            full_key = full_key+"_"+part_key

            # Load from RAM cache
            if not force_reload \
                    and cached_ram and full_key in ninja_globals["cache"]:
                if logger:
                    ninja_globals["logger"].debug(func.__name__+": Reading from RAM cache")
                return ninja_globals["cache"][full_key]

            # Resolve existence
            cache_file_default = os.path.join(ninja_globals["cache_dir"], full_key + ".pkl")
            exists = os.path.exists(cache_file_default) if check_fnc is None \
                else check_fnc(full_key, ninja_globals["cache_dir"])


            if not exists and \
                     dict_args.get("_write_to_cache", False) is False  and \
                    cache_google_cloud and ninja_globals["google_cache_on"]:

                if os.system(ninja_globals["gsutil_path"] + " stat "+os.path.join(ninja_globals["google_cloud_cache_dir"], full_key + "*")) == 0:
                    exists = True
                    ninja_globals["logger"].debug(func.__name__+": Reading from Google Cloud Storage")

                    os.system(ninja_globals["gsutil_path"] + " -m cp "+os.path.join(ninja_globals["google_cloud_cache_dir"], full_key + "* ") + \
                              ninja_globals["cache_dir"])


            def evaluate():
                if logger:
                    logger.debug(func.__name__+": Cache miss or force reload. Caching " + full_key)

                # Special function that can overwrite cache
                if dict_args.get("_write_to_cache", None) is not None:
                    returned_value = dict_args.get("_write_to_cache")
                else:
                    returned_value = func(*args, **dict_args_original)

                return returned_value

            def write(returned_value):
                if dict_args.get("store", True):
                    ninja_globals["logger"].debug(func.__name__+": Saving " + full_key)
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
                                                full_key, "search_args": dict({k: dict_args[k] for k in search_args}), \
                                                "args": json.loads(dumped_arguments)}))

                    if cache_google_cloud and ninja_globals["google_cache_on"]:
                        # if not os.system(ninja_globals["gsutil_path"] + " stat "+os.path.join(ninja_globals["google_cloud_cache_dir"], full_key + "*")) == 0:
                        os.system(ninja_globals["gsutil_path"]+" -m cp "+os.path.join(ninja_globals["cache_dir"], full_key + "* ") + " " +\
                                ninja_globals["google_cloud_cache_dir"])

                return returned_value
            # Load from cache unless some conditions are met
            if exists and not force_reload:

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
                       ninja_globals['logger'].info(func.__name__+":Corrupted file")
                       value = write(evaluate())

                if dict_args.get("_write_to_cache", False) != False:
                    # write(value)
                    return None # Just writing


            else:
                if not dict_args.get("_load_cache_or_fail", False):
                    value = write(evaluate())
                else:
                    return None

            if cached_ram:
                ninja_globals["cache"][full_key] = value

            return value

        if ninja_globals["cache_on"]:
            return func_caching
        else:
            return func

    return _cached


# This is an interesting hack that gives some key-value storage on top of kaggle_ninja
@cached(cache_google_cloud=True)
def _key_storage(**kwargs):
    raise ValueError("Not cached properly")

def ninja_get_value(**kwargs):
    return _key_storage(_load_cache_or_fail=True, **kwargs)

def ninja_set_value(value, **kwargs):
    _key_storage(_write_to_cache=value, **kwargs)


is_primitive = lambda v: isinstance(v, (int, float, bool, str))

def _validate_for_cached(x, skip_args=[], prefix=""):
    # Returns True/False if the x is primitive,list,dict, recursively

    if x is None:
        return x

    if is_primitive(x):
        return True
    elif isinstance(x, dict):
        if not all(isinstance(key, str) or isinstance(key, int) for key in x):
            return False
        return all(_validate_for_cached(v, skip_args=skip_args, prefix=prefix+str(k)+".") for k,v in x.iteritems()\
                  if prefix+str(k) not in skip_args\
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
        return dict({k: _clean_skipped_args(v, skip_args, prefix=prefix+str(k)+".") for k,v in x.iteritems() if prefix+str(k) not in skip_args})
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
    return t
