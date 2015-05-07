import unittest
from misc.config import c
from dummy_data_api import get_data, get_default_data_desc

class TestDummyDataApi(unittest.TestCase):

    def setUp(self):
        self.data_dir = c["DATA_DIR"]

    def test_default_data_desc(self):
        data_desc = get_default_data_desc()
        self.assertTrue(os.path.exists(data_desc['file']))

    def test_data
