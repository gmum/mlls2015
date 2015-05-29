import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(__file__))
import misc
from misc.config import *
from kaggle_ninja import *
from collections import namedtuple
import random_query
random_query_module = random_query

@cached(cache_google_cloud=True)
def run_experiment(name, **kwargs):
    # Note: this line might cause some problems with path. Add experiments folder to your path
    ex = __import__(name).ex
    ex.logger = get_logger(name)
    return ex.run(config_updates=kwargs).result
