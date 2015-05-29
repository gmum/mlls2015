import sys
sys.path.append("..")
import misc
from misc.config import *
from kaggle_ninja import *
from collections import namedtuple

ExperimentResults = namedtuple("ExperimentResults", ["results", "dumps", "monitors"])

