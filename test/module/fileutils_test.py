import os
import testutils
import fileutils
import pytest


def test_load_temp():
    temp_file = testutils.test_file(os.path.join('lammps_22Aug18','tmp.profile'))
    data = fileutils.load_temp(temp_file)
    assert (50, 4) == data.shape

@pytest.fixture
def energy_reader():
    ene_file = testutils.test_file(os.path.join('lammps_22Aug18', 'en_ex.log'))
    return fileutils.EnergyReader(ene_file)

def test_set_start_end(energy_reader):
    energy_reader.setStartEnd()
    assert 10 == energy_reader.start_line_num
    assert 50000 == energy_reader.thermo_intvl
    assert 400000000 == energy_reader.total_step_num

