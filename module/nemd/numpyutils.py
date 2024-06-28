import numpy as np


class Array(np.ndarray):
    """
    A subclass of numpy.ndarray that supports mapping the indexes.
    """

    def imap(self, index):
        if isinstance(index, slice):
            args = [index.start, index.stop, index.step]
            return slice(*[x if x is None else self.id_map[x] for x in args])
        # int, list, or np.ndarray
        return self.id_map[index]

    def __getitem__(self, index):
        nindex = tuple(self.imap(x) for x in index)
        data = super(Array, self).__getitem__(nindex)
        return np.asarray(data)

    def __setitem__(self, index, value):
        nindex = tuple(self.imap(x) for x in index)
        super(Array, self).__setitem__(nindex, value)


class TypeMap(np.ndarray):
    """
    A subclass of numpy.ndarray that represents a set of integers as a bit array.
    """

    def __new__(cls, size, dtype=int):
        array = np.zeros(size, dtype=dtype)
        obj = np.asarray(array).view(cls)
        return obj

    def union(self, indexes):
        self[indexes] = 1
        self[self.indexes] = np.arange(1, len(self.indexes) + 1)

    @property
    def indexes(self):
        return self.nonzero()[0]
