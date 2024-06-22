import os
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
        options = get_options()
        return lammpsin.In(options=options)

    def testResetFilenames(self, lmp_in):
        assert lmp_in.lammps_in == 'test.in'
        assert lmp_in.datafile == 'test.data'
        assert lmp_in.lammps_dump == 'test.custom.gz'
        lmp_in.resetFilenames(jobname='new_test')
        assert lmp_in.lammps_in == 'new_test.in'
        assert lmp_in.datafile == 'new_test.data'
        assert lmp_in.lammps_dump == 'new_test.custom.gz'

    # def writeDescriptions(self, lmp_in, tmp_path):
    #     with fileutils.chdir(tmp_path, rmtree=True):

    def testWriteIn(self, lmp_in, tmp_dir):
        lmp_in.writeIn()
        assert os.path.exists('test.in')