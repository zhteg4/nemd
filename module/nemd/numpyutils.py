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


class BitSet(np.ndarray):
    """
    A subclass of numpy.ndarray that represents integer list as a bit array.
    """

    def __new__(cls, max_val=0, dtype=int):
        """
        Create a new BitSet object.

        :param max_val: The maximum value of the bit array.
        :param dtype: The data type of the bit array.
        """
        array = np.zeros(max_val + 1, dtype=dtype)
        obj = np.asarray(array).view(cls)
        return obj

    def add(self, indexes):
        """
        Add bits at the given indexes.

        :param indexes: An iterable of indexes to add.
        """
        self[indexes] = True
        self[self.on] = np.arange(1, len(self.on) + 1)

    @property
    def on(self):
        """
        Return the indexes of the on bits.
        """
        return self.nonzero()[0]
