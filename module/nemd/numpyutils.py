import numpy as np


class Array(np.ndarray):

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
