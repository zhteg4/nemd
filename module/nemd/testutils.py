import pathlib
import os

TEST = 'test'
TEST_FIELS = 'test_files'
CUR_FILE_DIR = pathlib.Path(__file__).parent.absolute()
TEST_FILE_DIR = os.path.join(CUR_FILE_DIR, os.pardir, os.pardir, TEST,
                             TEST_FIELS)

SINGLE_NEMD = 'lammps_22Aug18/polyacetylene/single_chain/nemd'
CRYSTAL_NEMD = 'lammps_22Aug18/polyacetylene/crystal_cell/nemd'


def test_file(filename):
    return os.path.join(TEST_FILE_DIR, filename)
