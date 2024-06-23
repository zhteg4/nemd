import os
import io
import pytest

from nemd import lammpsin
from nemd import parserutils


def get_options(args=None):
    if args is None:
        args = []
    parser = parserutils.get_parser()
    parserutils.add_md_arguments(parser)
    parserutils.add_job_arguments(parser, jobname='test')
    return parser.parse_args(args)


class TestIn:

    @pytest.fixture
    def lmp_in(self):
        lmp_in = lammpsin.In(options=get_options())
        lmp_in.in_fh = io.StringIO()
        return lmp_in

    @staticmethod
    def getContents(lmp_in):
        lmp_in.in_fh.seek(0)
        return lmp_in.in_fh.read()

    def testSetFilenames(self, lmp_in):
        assert lmp_in.lammps_in == 'test.in'
        assert lmp_in.datafile == 'test.data'
        assert lmp_in.lammps_dump == 'test.custom.gz'
        lmp_in.setFilenames(jobname='new_test')
        assert lmp_in.lammps_in == 'new_test.in'
        assert lmp_in.datafile == 'new_test.data'
        assert lmp_in.lammps_dump == 'new_test.custom.gz'

    def testWriteSetup(self, tmp_dir, lmp_in):
        lmp_in.writeSetup()
        assert 'units real' in self.getContents(lmp_in)

    def testReadData(self, tmp_dir, lmp_in):
        lmp_in.readData()
        assert 'read_data test.data' in self.getContents(lmp_in)

    def testWriteMinimize(self, tmp_dir, lmp_in):
        lmp_in.writeMinimize()
        assert 'minimize' in self.getContents(lmp_in)

    def testWriteTimestep(self, tmp_dir, lmp_in):
        lmp_in.writeTimestep()
        assert 'timestep' in self.getContents(lmp_in)

    def testWriteRun(self, tmp_dir, lmp_in):
        lmp_in.writeRun()
        assert 'velocity' in self.getContents(lmp_in)

    def testWriteIn(self, tmp_dir, lmp_in):
        lmp_in.writeIn()
        assert os.path.exists('test.in')