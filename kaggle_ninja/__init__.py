#@ktos_madry: jak to zrobic ladnie?
ninja_globals = {"force_reload_all": False, "cache_on": True, "logger": None, "cache_dir": ".", "cache": {}, "register": {}}

def turn_off_cache():
    global ninja_globals
    ninja_globals["cache_on"] = False

def turn_off_cache():
    global ninja_globals
    ninja_globals["cache_on"] = True

def turn_on_force_reload_all():
    global ninja_globals
    ninja_globals["force_reload_all"] = True

def turn_off_force_reload_all():
    global ninja_globals
    ninja_globals["force_reload_all"] = False


def setup_ninja(logger, cache_dir):
    global ninja_globals
    ninja_globals["logger"] = logger
    ninja_globals["cache_dir"] = cache_dir

    
from .utils import timed, find_obj, register
from .cached import cached
from .cached_helpers import *


