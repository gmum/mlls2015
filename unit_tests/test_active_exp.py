import unittest
import sys
sys.path.append("..")
import kaggle_ninja
kaggle_ninja.turn_off_cache()

from misc.utils import *
from misc.config import c


class TestActiveExperiment(unittest.TestCase):

    def setUp(self):
        self.comps = [['5ht7', 'ExtFP']]
