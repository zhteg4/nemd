import os
import io
import types
import pytest

from nemd import lammpsin
from nemd import parserutils


class Base:

    @staticmethod
    def getOptions(args=None):
        if args is None:
            args = []
        parser = parserutils.get_parser()
        parserutils.add_md_arguments(parser)
        parserutils.add_job_arguments(parser, jobname='test')
        return parser.parse_args(args)

    @staticmethod
    def getContents(obj):
        obj.fh.seek(0)
        return obj.fh.read()


class TestIn(Base):

    @pytest.fixture
    def lmp_in(self):
        lmp_in = lammpsin.In(options=self.getOptions())
        lmp_in.fh = io.StringIO()
        return lmp_in

    def testSetFilenames(self, lmp_in):
        assert lmp_in.lammps_in == 'test.in'
        assert lmp_in.datafile == 'test.data'
        assert lmp_in.lammps_dump == 'test.custom.gz'
        lmp_in.setFilenames(jobname='new_test')
        assert lmp_in.lammps_in == 'new_test.in'
        assert lmp_in.datafile == 'new_test.data'
        assert lmp_in.lammps_dump == 'new_test.custom.gz'

    def testWriteSetup(self, lmp_in):
        lmp_in.writeSetup()
        assert 'units real' in self.getContents(lmp_in)

    def testReadData(self, lmp_in):
        lmp_in.readData()
        assert 'read_data test.data' in self.getContents(lmp_in)

    def testWriteMinimize(self, lmp_in):
        lmp_in.writeMinimize()
        assert 'minimize' in self.getContents(lmp_in)

    def testWriteTimestep(self, lmp_in):
        lmp_in.writeTimestep()
        assert 'timestep' in self.getContents(lmp_in)

    def testWriteRun(self, lmp_in):
        lmp_in.writeRun()
        assert 'velocity' in self.getContents(lmp_in)

    def testWriteIn(self, tmp_dir, lmp_in):
        lmp_in.writeIn()
        assert os.path.exists('test.in')


class TestFixWriter(Base):

    @pytest.fixture
    def fix_writer(self):
        options = {x: y for x, y in self.getOptions()._get_kwargs()}
        struct_info = types.SimpleNamespace(btypes=[2],
                                            atypes=[1],
                                            testing=False)
        options = types.SimpleNamespace(**options, **struct_info.__dict__)
        return lammpsin.FixWriter(io.StringIO(), options=options)

    @staticmethod
    def getContents(obj):
        obj.write()
        return super(TestFixWriter, TestFixWriter).getContents(obj)

    def testWriteFix(self, fix_writer):
        fix_writer.fixShake()
        assert 'b 2 a 1' in self.getContents(fix_writer)

    def testVelocity(self, fix_writer):
        fix_writer.velocity()
        assert 'create' in self.getContents(fix_writer)

    def testStartLow(self, fix_writer):
        fix_writer.startLow()
        assert 'temp/berendsen' in self.getContents(fix_writer)

    def testRampUp(self, fix_writer):
        fix_writer.rampUp()
        assert 'loop' in self.getContents(fix_writer)

    @pytest.mark.parametrize('prod_ens, args', [('NVT', True), ('NPT', False)])
    def testRelaxAndDefrom(self, fix_writer, prod_ens, args):
        fix_writer.options.prod_ens = prod_ens
        fix_writer.relaxAndDefrom()
        contain = 'change_box' in self.getContents(fix_writer)
        assert contain == args

    @pytest.mark.parametrize('prod_ens, args',
                             [('NVE', 'nve'), ('NVT', 'temp'),
                              (None, 'press')])
    def testProduction(self, fix_writer, prod_ens, args):
        fix_writer.options.prod_ens = prod_ens
        fix_writer.production()
        assert args in self.getContents(fix_writer)
