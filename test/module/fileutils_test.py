import os
import testutils
import fileutils

def test_load_temp_file():
    temp_file = testutils.test_file(os.path.join('lammps_22Aug18','tmp.profile'))
    data = fileutils.load_temp_file(temp_file)
    assert (50, 4) == data.shape
