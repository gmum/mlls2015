import numpy as np


class ObstructedY(object):
    
    def __init__(self, y):
        self._y = np.array(y)
        self.size = len(y)
        self.shape = y.shape
        self.known = np.array([False for _ in xrange(self.size)])
        self.unknown_ids = np.where(self.known == False)[0]
        self.classes = np.unique(self._y)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key > self.size :
                raise IndexError, "Index %i out of bound for array of size %i" % (key, self.size)

            if not self.known[key]:
                raise IndexError, "Element at index %i needs to be queried before accessing it!" % key

            return self._y[key]
        elif isinstance(key, slice):
            if max([key.start, key.stop, key.step]) >= self.size:
                raise IndexError, "Index out of bound"
            if not all(self.known[key]):
                raise IndexError, "All elements need to be queried before accessing them!"
            return self._y[key]
        elif isinstance(key, list):
            if not all (self.known[key]):
                raise IndexError, "All elements need to be queried before accessing them!"
            return self._y[key]
        elif isinstance(key, np.ndarray):
            if not all (self.known[key]):
                raise IndexError, "All elements need to be queried before accessing them!"
            return self._y[key]

        else:
            raise TypeError, "Invalid argument type"

    def query(self, ind):
        self.known[ind] = True
        self.unknown_ids = np.where(self.known == False)[0]
        return self._y[ind]

    def peek(self):
        return self._y[np.invert(self.known)]

