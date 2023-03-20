import os
import pytest
from nemd import testutils
from nemd import fileutils
from nemd.testutils import SINGLE_NEMD, CRYSTAL_NEMD


class TestTempReader(object):
    @pytest.fixture
    def temp_reader(self):
        temp_file = testutils.test_file(
            os.path.join(SINGLE_NEMD, 'temp.profile'))
        temp_reader = fileutils.TempReader(temp_file)
        return temp_reader

    def testRun(self, temp_reader):
        temp_reader.run()
        assert (50, 4, 6) == temp_reader.data.shape


class TestEnergyReader(object):
    @pytest.fixture
    def energy_reader(self):
        ene_file = testutils.test_file(os.path.join(SINGLE_NEMD, 'en_ex.log'))
        return fileutils.EnergyReader(ene_file, 0.25)

    def testSetStartEnd(self, energy_reader):
        energy_reader.setStartEnd()
        assert 10 == energy_reader.start_line_num
        assert 50000 == energy_reader.thermo_intvl
        assert 400000000 == energy_reader.total_step_num


class TestLammpsInput(object):
    @pytest.fixture
    def lammps_input_reader(self):
        input_file = testutils.test_file(
            os.path.join(SINGLE_NEMD, 'in.nemd_cff91'))
        lammps_in = fileutils.LammpsInput(input_file)
        return lammps_in

    def testRun(self, lammps_input_reader):
        lammps_input_reader.run()
        'real' == lammps_input_reader.cmd_items['units']
        'full' == lammps_input_reader.cmd_items['atom_style']
        '*' == lammps_input_reader.cmd_items['processors'].x
        1 == lammps_input_reader.cmd_items['processors'].y


class TestLammpsLogReader(object):
    @pytest.fixture
    def lammps_log_reader(self):
        log_file = testutils.test_file(os.path.join(CRYSTAL_NEMD,
                                                    'log.lammps'))
        lammps_log = fileutils.LammpsLogReader(log_file)
        return lammps_log

    def testRun(self, lammps_log_reader):
        lammps_log_reader.run()
        assert 6 == len(lammps_log_reader.all_data)
