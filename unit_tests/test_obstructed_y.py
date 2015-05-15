# TODO: test np.ndarray as a key

import unittest
import numpy as np
from models.utils import ObstructedY


class TestObstructedY(unittest.TestCase):

    def setUp(self):
        self.y = np.ones(100)
        self.y[np.random.randint(0, 100)] = -1

    def test_constructor(self):
        oy = ObstructedY(self.y)

        self.assertTrue(all(oy._y == self.y))
        self.assertTrue(not any(oy.known))
        self.assertEqual(len(self.y[oy.known]), 0)

    def test_element_access(self):
        oy = ObstructedY(self.y)

        self.assertEqual(oy.query(42), self.y[42])
        self.assertEqual(oy[42], self.y[42])

        self.assertTrue(all(oy.query([6,66]) == self.y[[6,66]]))
        self.assertTrue(all(oy[[6,66]] == self.y[[6,66]]))

        oy.query([3,4,5,6])
        self.assertTrue(all(oy[3:7] == self.y[3:7]))

    def test_full_query(self):
        oy = ObstructedY(self.y)
        self.assertTrue(all(oy.query(range(100)) == self.y))
        self.assertTrue(all(oy[:] == self.y))
        self.assertTrue(all(oy.known))

    @unittest.expectedFailure
    def test_nad_index_single(self):
        oy = ObstructedY(self.y)
        oy[666]

    @unittest.expectedFailure
    def test_nad_index_slice(self):
        oy = ObstructedY(self.y)
        oy[666:777]

    @unittest.expectedFailure
    def test_nad_index_list(self):
        oy = ObstructedY(self.y)
        oy[[666,777]]

    @unittest.expectedFailure
    def test_bad_access_single(self):
        oy = ObstructedY(self.y)
        oy[42]

    @unittest.expectedFailure
    def test_bad_access_slice(self):
        oy = ObstructedY(self.y)
        oy[6:66]

    @unittest.expectedFailure
    def test_bad_access_list(self):
        oy = ObstructedY(self.y)
        oy[[6,66]]


suite = unittest.TestLoader().loadTestsFromTestCase(TestObstructedY)
print unittest.TextTestRunner(verbosity=3).run(suite)