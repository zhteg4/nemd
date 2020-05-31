import os
import testutils
import fileutils
import pytest

SINGLE_NEMD = 'lammps_22Aug18/polyacetylene/single_chain/nemd'
def test_load_temp():
    temp_file = testutils.test_file(os.path.join(SINGLE_NEMD, 'temp.profile'))
    data, frame_num = fileutils.load_temp(temp_file)
    assert (50, 4, 6) == data.shape

@pytest.fixture
def energy_reader():
    ene_file = testutils.test_file(os.path.join(SINGLE_NEMD, 'en_ex.log'))
    return fileutils.EnergyReader(ene_file, 0.25)

def test_set_start_end(energy_reader):
    energy_reader.setStartEnd()
    assert 10 == energy_reader.start_line_num
    assert 50000 == energy_reader.thermo_intvl
    assert 400000000 == energy_reader.total_step_num

@pytest.fixture
def lammps_input_reader():
    input_file = testutils.test_file(os.path.join(SINGLE_NEMD, 'in.nemd_cff91'))
    lammps_in = fileutils.LammpsInput(input_file)
    return lammps_in

def test_run(lammps_input_reader):
    lammps_input_reader.run()
    'real' == lammps_input_reader.cmd_items['units']
    'full' == lammps_input_reader.cmd_items['atom_style']
    '*' == lammps_input_reader.cmd_items['processors'].x
    1 == lammps_input_reader.cmd_items['processors'].y
