import sys
sys.path.append("..")
import misc
from misc.config import *
from kaggle_ninja import *
from collections import namedtuple

ExperimentResults = namedtuple("ExperimentResults", ["results", "dumps", "monitors"])

def run_experiment(ex, **kwargs):
    # Note: this line might cause some problems with path. Add experiments folder to your path
    ex.logger = get_logger(ex.name)
    return ex.run(config_updates=kwargs).result