import re
import logging
import multiprocessing
from multiprocessing.pool import Pool # subprocess will clean itself allocated memory - convenients


ninja_globals = {"gsutil_path": "gsutil", "slave_pool": None, "current_tasks": [],
                 "force_reload": set(), "google_cache_on": False, "google_cloud_cache_dir": "",\
                 "cache_on": True, "logger": logging.getLogger("kaggle_ninja"), "cache_dir": ".", "cache": {}, "register": {}}


def turn_off_cache():
    global ninja_globals
    ninja_globals["cache_on"] = False

def turn_off_cache():
    global ninja_globals
    ninja_globals["cache_on"] = True

def turn_on_force_reload_all():
    global ninja_globals
    ninja_globals["force_reload"].add(".*")

def turn_off_force_reload_all():
    global ninja_globals
    ninja_globals["force_reload"].remove(".*")

def turn_on_force_reload(func_name):
    assert len(func_name) > 0
    global ninja_globals
    ninja_globals["force_reload"].add(func_name)

def turn_off_force_reload(func_name):
    assert len(func_name) > 0
    global ninja_globals
    ninja_globals["force_reload"].remove(func_name)

def setup_ninja(logger, cache_dir, google_cloud_cache_dir="", gsutil_path="gsutil"):
    global ninja_globals
    ninja_globals["logger"] = logger
    ninja_globals["gsutil_path"] = gsutil_path
    ninja_globals["cache_dir"] = cache_dir
    ninja_globals["google_cloud_cache_dir"] = google_cloud_cache_dir
    if google_cloud_cache_dir != "":
        ninja_globals["google_cache_on"] = True

from .utils import timed, find_obj, register
from .cached import cached, get_last_cached, get_all_by_search_arg, ninja_get_value, ninja_set_value
from .cached_helpers import *
from .cached_helpers import mmap_joblib_load, mmap_numpy_load_fnc
from .parallel_computing import  *



