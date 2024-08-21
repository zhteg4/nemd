import os
import sys
import pytest

from nemd import fileutils
from nemd import environutils

SINGLE_NEMD = 'lammps_22Aug18/polyacetylene/single_chain/nemd'
CRYSTAL_NEMD = 'lammps_22Aug18/polyacetylene/crystal_cell/nemd'

TEST_DIR = environutils.get_test_dir()
if TEST_DIR is None:
    sys.exit('Error: cannot find test directory')
FILES_DIR = os.path.join(TEST_DIR, 'test_files')


class TestTempReader(object):

    @pytest.fixture
    def temp_reader(self):
        temp_file = os.path.join(FILES_DIR, SINGLE_NEMD, 'temp.profile')
        temp_reader = fileutils.TempReader(temp_file)
        return temp_reader

    def testRun(self, temp_reader):
        temp_reader.run()
        assert (50, 4, 6) == temp_reader.data.shape


class TestEnergyReader(object):

    @pytest.fixture
    def energy_reader(self):
        ene_file = os.path.join(FILES_DIR, SINGLE_NEMD, 'en_ex.log')
        return fileutils.EnergyReader(ene_file, 0.25)

    def testSetStartEnd(self, energy_reader):
        energy_reader.setStartEnd()
        assert 10 == energy_reader.start_line_num
        assert 50000 == energy_reader.thermo_intvl
        assert 400000000 == energy_reader.total_step_num


class TestLammpsInput(object):

    @pytest.fixture
    def inscriptput_reader(self):
        input_file = os.path.join(FILES_DIR, SINGLE_NEMD, 'in.nemd_cff91')
        inscript = fileutils.LammpsInput(input_file)
        return inscript

    def testRun(self, inscriptput_reader):
        inscriptput_reader.run()
        'real' == inscriptput_reader.cmd_items['units']
        'full' == inscriptput_reader.cmd_items['atom_style']
        '*' == inscriptput_reader.cmd_items['processors'].x
        1 == inscriptput_reader.cmd_items['processors'].y


class TestLammpsLogReader(object):

    @pytest.fixture
    def lammps_log_reader(self):
        log_file = os.path.join(FILES_DIR, CRYSTAL_NEMD, 'log.lammps')
        lammps_log = fileutils.LammpsLogReader(log_file)
        return lammps_log

    def testRun(self, lammps_log_reader):
        lammps_log_reader.run()
        assert 6 == len(lammps_log_reader.all_data)
