# Whole architecture test on [-1,-1]^2 dataset

import sys
sys.path.append("..")
import kaggle_ninja
from kaggle_ninja import *
import random_query, random_query_composite
from experiments import experiment_runner, fit_active_learning, fit_grid
from experiment_runner import run_experiment, run_experiment_grid
from experiments.utils import plot_grid_experiment_results, get_best
from misc.config import *
# from experiment_runner import _replace_in_json


import unittest
import os
