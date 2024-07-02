import pytest
from nemd import numpyutils


class TestBitSet:

    @pytest.fixture
    def bitset(self):
        return numpyutils.BitSet(10)

    def testAdd(self, bitset):
        assert not bitset[3]
        bitset.add(3)
        assert bitset[3]

    def testOn(self, bitset):
        assert not bitset.on.any()
        bitset.add(6)
        assert (bitset.on == 6).all()
