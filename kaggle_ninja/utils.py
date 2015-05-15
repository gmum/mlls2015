import time

from kaggle_ninja import ninja_globals

def register(obj_name, obj):
    """
    To my best knowledge there is no other way to do it? 
    """
    ninja_globals['register'][obj_name] = obj

def find_obj(obj_name):
    global ninja_globals
    if not obj_name in ninja_globals['register']:
        if not obj_name in globals():
            raise ValueError("Didn't find "+obj_name)
        else:
            return globals()[obj_name]
    return ninja_globals['register'][obj_name]

def timed(func, logger=None):
    """ Decorator for easy time measurement """
    
    def timed(*args, **dict_args):
        tstart = time.time()
        result = func(*args, **dict_args)
        if logger:
            logger.info("{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), time.time() - tstart))
        else:
            print("{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), time.time() - tstart))
            
        return result

    return timed