import sys
import logutils
import os
import units
import parserutils
import fileutils
import nemd
import plotutils
import environutils
import jobutils
import numpy as np


NUM_REPEAT_UNIT = 'num_repeat_unit'
Y_CELL_SIZE = '-y_cell_size'
Z_CELL_SIZE = '-z_cell_size'


JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


def log_debug(msg):
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


class Polyacetylene(object):

    ATOM_PER_REPEAT_UNIT = 2
    CHAIN_PER_CELL = 2

    Cell_Size_X = 2.46775755
    Cell_Size_Y = 7.380
    Cell_Size_Z = 4.120

    Chain1_XYZ = np.array([[-0.602, 0.602], [0.3025, -0.3025], [0.2580, -0.2580]]) + 3
    Chain2_XYZ = np.array([[-0.602, 0.602], [3.3835, 3.9885], [-2.2070, -2.7230]]) + 3

    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.num_repeat_unit = self.options.num_repeat_unit
        self.y_cell_size = self.options.y_cell_size
        self.z_cell_size = self.options.z_cell_size
        self.outfile = self.jobname + '.lammps'
        self.total_atom = self.y_cell_size * self.z_cell_size * self.ATOM_PER_REPEAT_UNIT * self.CHAIN_PER_CELL

    def run(self):
        with open(self.outfile, 'w') as fh_lammp:
            fh_lammp.write(self.getInteractionType())

    def getInteractionType(self):
        header = []
        header.append('Lammps Data Files By Teng \n')
        header.append('\n')
        header.append('%i atoms \n' % self.total_atom)
        header.append('%i bonds \n' % self.total_atom)
        header.append('%i angles \n' % self.total_atom)
        header.append('%i dihedrals \n' % self.total_atom)
        header.append('%i impropers \n' % 0)
        header.append('\n')

fh = open('./dataCrystal3.lammps', 'w')

fh.write('Lammps Data Files By Teng \n')
fh.write('\n')
fh.write('%i atoms \n' % NumAtom)
fh.write('%i bonds \n' % NumAtom)
fh.write('%i angles \n' % NumAtom)
fh.write('%i dihedrals \n' % NumAtom)
fh.write('%i impropers \n' % 0)

fh.write('\n')
fh.write('%i atom types \n' % 1)
fh.write('%i bond types \n' % 2)
fh.write('%i angle types \n' % 1)
fh.write('%i dihedral types \n' % 2)

fh.write('\n')
fh.write(f'0 {supX * CellSizeX:.6f}  xlo xhi \n')
fh.write(f'0 {ny * CellSizeY:.6f}  ylo yhi \n')
fh.write(f'0 {nz * CellSizeZ:.6f}  zlo zhi \n')

fh.write('\n')
fh.write('Masses \n')
fh.write('\n')
fh.write(' 1  13.0191 \n')

fh.write('\n')
fh.write('Pair Coeffs \n')
fh.write('\n')
fh.write(' 1   0.06400   4.010000 \n')

fh.write('\n')
fh.write('Bond Coeffs \n')
fh.write('\n')
fh.write(' 1     128.263	2.068	1.368 \n')
fh.write(' 2     100.025	2.032	1.421 \n')

fh.write('\n')
fh.write('Angle Coeffs \n')
fh.write('\n')
fh.write(' 1   124.5   83.83	-52.26	23.31 \n')

fh.write('\n')
fh.write('Dihedral Coeffs \n')
fh.write('\n')
fh.write(' 1     27.48	-1.32	-49.16	2.54	23.37 \n')
fh.write(' 2     12.70	-1.95	-15.61	3.77	4.68  \n')

fh.write('\n')
fh.write('Atoms \n')
fh.write('\n')
iatom = 0
chainID = 0

for j in range(1, ny + 1):
    for k in range(1, nz + 1):
        chainID = chainID + 1
        for i in range(1, supX + 1):
            iatom = iatom + 1
            fh.write(
                f" {iatom} {chainID} 1 0 {Chain1[0, 0] + CellSizeX * (i - 1):.6f} "
                f"{Chain1[1, 0] + CellSizeY * (j - 1):.6f} {Chain1[2, 0] + CellSizeZ * (k - 1):.6f}\n"
            )
            iatom = iatom + 1
            fh.write(
                f" {iatom} {chainID} 1 0 {Chain1[0, 1] + CellSizeX * (i - 1):.6f} "
                f"{Chain1[1,1] + CellSizeY * (j - 1):.6f} {Chain1[2,1] + CellSizeZ * (k - 1):.6f}\n"
            )

        chainID = chainID + 1

        for i in range(supX + 1, 1):
            iatom = iatom + 1
            fh.write(
                f" {iatom} {chainID} 1 0 {Chain2[0,0] + CellSizeX * (i - 1)}:f.6"
                f"{Chain2[2, 1] + CellSizeY * (j - 1):f.6} {Chain2[2, 0] + CellSizeZ * (k - 1):f.6}\n"
            )
            iatom = iatom + 1
            fh.write(
                f" {iatom} {chainID} 1 0 {Chain2[1, 2] + CellSizeX * (i - 1):f.6} "
                f"{Chain2(2, 2) + CellSizeY * (j - 1):f.6} {Chain2(3, 2) + CellSizeZ * (k - 1):f.6}\n"
            )

fh.write('\n')
fh.write('Bonds \n')
fh.write('\n')

ibondStart = 1

for i in range(1, N +1):
    TotalAtom=supX*NumAtomUnitCell;
    ibond=ibondStart;
    ibond1=ibondStart;
    ibond2=ibondStart+1;
    SupperLimit=ibondStart+TotalAtom-1;
    for i in range(1, supX + 1):
        if ibond2 > SupperLimit:
            ibond2= ibond2-TotalAtom;

        fh.write(f'{ibond} 1 {ibond1} {ibond2}\n')
        ibond=ibond+1;
        ibond1=ibond1+1;
        ibond2=ibond2+1;

        if ibond2 > SupperLimit:
            ibond2= ibond2-TotalAtom

        fh.write(f"{ibond} 2 {ibond1} {ibond2}\n")
        ibond=ibond+1;
        ibond1=ibond1+1;
        ibond2=ibond2+1;
    ibondStart = ibond


fh.write( '\n');
fh.write( 'Angles \n');
fh.write( '\n');
iangleStart = 1;

for i in range(1, N + 1):
    TotalAtom = supX * NumAtomUnitCell;
    iangle = iangleStart;
    iangle1 = iangleStart;
    iangle2 = iangleStart + 1;
    iangle3 = iangleStart + 2;
    SupperLimit = iangleStart + TotalAtom - 1;
    for i in range(1, TotalAtom + 1):
        if iangle2 > SupperLimit:
            iangle2 = iangle2 - TotalAtom

        if iangle3 > SupperLimit:
            iangle3 = iangle3 - TotalAtom

        fh.write(f"{iangle} 1 {iangle1} {iangle2} {iangle3}\n")
        iangle = iangle + 1
        iangle1 = iangle1 + 1
        iangle2 = iangle2 + 1
        iangle3 = iangle3 + 1
    iangleStart = iangle

fh.write( '\n');
fh.write( 'Dihedrals \n');
fh.write( '\n');
idiheStart = 1;

for i in range(1, N +1):
    TotalAtom = supX * NumAtomUnitCell;
    idihe = idiheStart;
    idihe1 = idiheStart;
    idihe2 = idiheStart + 1;
    idihe3 = idiheStart + 2;
    idihe4 = idiheStart + 3;
    SupperLimit = idiheStart + TotalAtom - 1;
    for i in range(1, supX +1):
        if idihe2 > SupperLimit:
            idihe2 = idihe2 - TotalAtom;

        if idihe3 > SupperLimit:
            idihe3 = idihe3 - TotalAtom;

        if idihe4 > SupperLimit:
            idihe4 = idihe4 - TotalAtom;

        fh.write(f"{idihe} 2 {idihe1} {idihe2} {idihe3} {idihe4}\n")
        idihe = idihe + 1;
        idihe1 = idihe;
        idihe2 = idihe + 1;
        idihe3 = idihe + 2;
        idihe4 = idihe + 3;

        if idihe2 > SupperLimit:
            idihe2 = idihe2 - TotalAtom;

        if idihe3 > SupperLimit:
            idihe3 = idihe3 - TotalAtom;

        if idihe4 > SupperLimit:
            idihe4 = idihe4 - TotalAtom;

        fh.write(f"{idihe} 1 {idihe1} {idihe2} {idihe3} {idihe4}\n")
        idihe = idihe + 1;
        idihe1 = idihe;
        idihe2 = idihe + 1;
        idihe3 = idihe + 2;
        idihe4 = idihe + 3;

    idiheStart = idihe

fh.close()

def get_parser():
    parser = parserutils.get_parser(
        description=
        'Calculate thermal conductivity using non-equilibrium molecular dynamics.'
    )
    parser.add_argument(NUM_REPEAT_UNIT,
                        metavar='INT',
                        type=parserutils.type_positive_int,
                        help='')
    parser.add_argument(Y_CELL_SIZE,
                        metavar='INT',
                        type=parserutils.type_positive_int,
                        default=4,
                        help='')
    parser.add_argument(Z_CELL_SIZE,
                        metavar='INT',
                        type=parserutils.type_positive_int,
                        default=6,
                        help='')
    jobutils.add_job_arguments(parser)
    return parser


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    nemd = Polyacetylene(options, jobname)
    nemd.run()


if __name__ == "__main__":
    main(sys.argv[1:])
