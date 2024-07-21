import numpy as np


class Array(np.ndarray):
    """
    A subclass of numpy.ndarray that supports mapping the indexes.
    """

    def imap(self, index):
        """
        Map the given index to the corresponding index in the array.

        :param index: the input index.
        :type index: int, list, np.ndarray, or slice
        :return: the mapped index.
        :rtype: int, list, np.ndarray, or slice
        """
        if isinstance(index, slice):
            args = [index.start, index.stop, index.step]
            return slice(*[x if x is None else self.id_map[x] for x in args])
        # int, list, or np.ndarray
        return self.id_map[index]

    def __getitem__(self, index):
        """
        Get the item(s) at the given index(es).

        :param index: the input index.
        :type index: int, list, np.ndarray, or slice
        :return: the item(s) at the given index(es)
        :rtype: int or np.ndarray
        """
        nindex = tuple(self.imap(x) for x in index)
        data = super(Array, self).__getitem__(nindex)
        return np.asarray(data)

    def __setitem__(self, index, value):
        """
        Set the item(s) at the given index(es) to the given value(s).

        :param index: the input index.
        :type index: int, list, np.ndarray, or slice
        :param value: the value to set.
        :type value: int, list, np.ndarray
        """
        nindex = tuple(self.imap(x) for x in index)
        super(Array, self).__setitem__(nindex, value)


class IntArray(np.ndarray):
    """
    A subclass of numpy.ndarray that represents integer list as an array for
    nonzero indexing.
    """

    def __new__(cls, max_val=0, dtype=bool):
        """
        Create a new BitSet object.

        :param max_val: The maximum value of the bit array.
        :param dtype: The data type of the bit array.
        """
        array = np.zeros(max_val + 1, dtype=dtype)
        obj = np.asarray(array).view(cls)
        return obj

    @property
    def on(self):
        """
        Return the indexes of the on bits.
        """
        return self.nonzero()[0]

    def map(self, ids):
        """
        Map the given indexes to the range of the on bits.

        :param ids: An iterable of indexes to add.
        :return np.ndarray: the mapped indexes.
        :raise KeyError: if any of the indexes is not in the on bits.
        """
        imap = {x: y for x, y in zip(self.on, range(self.sum()))}
        try:
            return np.array([imap[x] for x in ids])
        except KeyError:
            raise KeyError(f"{ids} not in {imap.keys()}")
