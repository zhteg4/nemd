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

FLAG_NUM_REPEAT_UNIT = 'num_repeat_unit'
FLAG_Y_CELL_NUM = '-y_cell_num'
FLAG_Z_CELL_NUM = '-z_cell_num'

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


def log_debug(msg):
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


class Polyacetylene(object):

    ATOM_PER_FRAGMENT = 2
    CHAIN_PER_CELL = 2

    CELL_X = 2.46775755
    CELL_Y = 7.380
    CELL_Z = 4.120

    CHAIN1_XYZ = np.array([[2.398, 3.602], [3.3025, 2.6975], [3.258, 2.742]])
    CHAIN2_XYZ = np.array([[2.398, 3.602], [6.3835, 6.9885], [0.793, 0.277]])

    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.num_repeat_unit = self.options.num_repeat_unit
        self.y_cell_num = self.options.y_cell_num
        self.z_cell_num = self.options.z_cell_num
        self.outfile = self.jobname + '.lammps'
        self.chain_num = self.y_cell_num * self.z_cell_num * self.CHAIN_PER_CELL
        self.atom_per_chain = self.num_repeat_unit * self.ATOM_PER_FRAGMENT
        self.total_atom = self.chain_num * self.atom_per_chain
        self.interaction_counts = []
        self.cell_sizes = []
        self.force_field = {}
        self.interactions = {}

    def run(self):
        self.setInteractionCounts()
        self.setCellSizes()
        self.setForceField()
        self.setInteractions()
        self.write()

    def write(self):
        with open(self.outfile, 'w') as fh_lammp:
            fh_lammp.writelines([f'{x}\n' for x in self.interaction_counts])
            fh_lammp.write('\n')
            fh_lammp.writelines([f'{x}\n' for x in self.cell_sizes])
            fh_lammp.write('\n')
            for header, lines in self.force_field.items():
                fh_lammp.write(f'{header}\n')
                fh_lammp.write('\n')
                fh_lammp.writelines([f' {line}\n' for line in lines])
                fh_lammp.write('\n')
            for header, lines in self.interactions.items():
                fh_lammp.write(f'{header}\n')
                fh_lammp.write('\n')
                fh_lammp.writelines([f' {line}\n' for line in lines])
                fh_lammp.write('\n')

    def setInteractionCounts(self):
        self.interaction_counts.append('Lammps Data Files By Teng')
        self.interaction_counts.append('')
        self.interaction_counts.append('%i atoms' % self.total_atom)
        self.interaction_counts.append('%i bonds' % self.total_atom)
        self.interaction_counts.append('%i angles' % self.total_atom)
        self.interaction_counts.append('%i dihedrals' % self.total_atom)
        self.interaction_counts.append('0 impropers')
        self.interaction_counts.append('')
        self.interaction_counts.append('1 atom types')
        self.interaction_counts.append('2 bond types')
        self.interaction_counts.append('1 angle types')
        self.interaction_counts.append('2 dihedral types')

    def setCellSizes(self):
        self.cell_sizes.append(
            f'0 {self.num_repeat_unit * self.CELL_X:.6f}  xlo xhi')
        self.cell_sizes.append(
            f'0 {self.y_cell_num * self.CELL_Y:.6f}  ylo yhi')
        self.cell_sizes.append(
            f'0 {self.z_cell_num * self.CELL_X:.6f}  zlo zhi')

    def setForceField(self):
        self.force_field['Masses'] = ['1  13.0191']
        self.force_field['Pair Coeffs'] = ['1   0.06400   4.010000']
        self.force_field['Bond Coeffs'] = [
            '1     128.263	2.068	1.368', '2     100.025	2.032	1.421'
        ]
        self.force_field['Angle Coeffs'] = ['1   124.5   83.83	-52.26	23.31']
        self.force_field['Dihedral Coeffs'] = [
            '1     27.48	-1.32	-49.16	2.54	23.37',
            '2     12.70	-1.95	-15.61	3.77	4.68'
        ]

    def setInteractions(self):
        self.interactions['Atoms'] = self.getAtoms()
        self.interactions['Bonds'] = self.getBonds()
        self.interactions['Angles'] = self.getAngles()
        self.interactions['Dihedrals'] = self.getDihedrals()

    def getAtoms(self):
        iatom = 0
        chain_id = 0
        atoms = []
        for y_cell_index in range(self.y_cell_num):
            for z_cell_index in range(self.z_cell_num):
                for chain_xyz in [self.CHAIN1_XYZ, self.CHAIN2_XYZ]:
                    chain_id = chain_id + 1
                    for repeat_unit_index in range(self.num_repeat_unit):
                        for unit_atom_id in range(2):
                            iatom = iatom + 1
                            x_coord = chain_xyz[
                                0,
                                unit_atom_id] + self.CELL_X * repeat_unit_index
                            y_coord = chain_xyz[
                                1, unit_atom_id] + self.CELL_Y * y_cell_index
                            z_coord = chain_xyz[
                                2, unit_atom_id] + self.CELL_Z * z_cell_index
                            atoms.append(
                                f" {iatom} {chain_id} 1 0 {x_coord:.6f} {y_coord:.6f} {z_coord:.6f}"
                            )
        return atoms

    def getBonds(self):
        bonds = []
        bond_id = 0
        for chain_id in range(self.chain_num):
            atom1_id = bond_id
            atom2_id = bond_id + 1
            max_atom_id = bond_id + self.atom_per_chain
            for repeat_unit_id in range(self.num_repeat_unit):
                for bond_type in range(1, 3):
                    bond_id = bond_id + 1
                    atom1_id = atom1_id + 1
                    atom2_id = atom2_id + 1
                    if atom2_id > max_atom_id:
                        atom2_id = atom2_id - self.atom_per_chain
                    bonds.append(
                        f'{bond_id} {bond_type} {atom1_id} {atom2_id}')

        return bonds

    def getAngles(self):

        angles = []
        iangleStart = 1
        for i in range(1, self.chain_num + 1):
            TotalAtom = self.num_repeat_unit * self.ATOM_PER_FRAGMENT
            iangle = iangleStart
            iangle1 = iangleStart
            iangle2 = iangleStart + 1
            iangle3 = iangleStart + 2
            SupperLimit = iangleStart + TotalAtom - 1
            for i in range(1, TotalAtom + 1):
                if iangle2 > SupperLimit:
                    iangle2 = iangle2 - TotalAtom

                if iangle3 > SupperLimit:
                    iangle3 = iangle3 - TotalAtom

                angles.append(f"{iangle} 1 {iangle1} {iangle2} {iangle3}")
                iangle = iangle + 1
                iangle1 = iangle1 + 1
                iangle2 = iangle2 + 1
                iangle3 = iangle3 + 1
            iangleStart = iangle
        return angles

    def getDihedrals(self):
        dihedrals = []
        idiheStart = 1
        for i in range(1, self.chain_num + 1):
            TotalAtom = self.num_repeat_unit * self.ATOM_PER_FRAGMENT
            idihe = idiheStart
            idihe1 = idiheStart
            idihe2 = idiheStart + 1
            idihe3 = idiheStart + 2
            idihe4 = idiheStart + 3
            SupperLimit = idiheStart + TotalAtom - 1
            for i in range(1, self.num_repeat_unit + 1):
                if idihe2 > SupperLimit:
                    idihe2 = idihe2 - TotalAtom

                if idihe3 > SupperLimit:
                    idihe3 = idihe3 - TotalAtom

                if idihe4 > SupperLimit:
                    idihe4 = idihe4 - TotalAtom

                dihedrals.append(
                    f"{idihe} 2 {idihe1} {idihe2} {idihe3} {idihe4}\n")
                idihe = idihe + 1
                idihe1 = idihe
                idihe2 = idihe + 1
                idihe3 = idihe + 2
                idihe4 = idihe + 3

                if idihe2 > SupperLimit:
                    idihe2 = idihe2 - TotalAtom

                if idihe3 > SupperLimit:
                    idihe3 = idihe3 - TotalAtom

                if idihe4 > SupperLimit:
                    idihe4 = idihe4 - TotalAtom

                dihedrals.append(
                    f"{idihe} 1 {idihe1} {idihe2} {idihe3} {idihe4}\n")
                idihe = idihe + 1
                idihe1 = idihe
                idihe2 = idihe + 1
                idihe3 = idihe + 2
                idihe4 = idihe + 3

            idiheStart = idihe
        return dihedrals


def get_parser():
    parser = parserutils.get_parser(
        description=
        'Calculate thermal conductivity using non-equilibrium molecular dynamics.'
    )
    parser.add_argument(FLAG_NUM_REPEAT_UNIT,
                        metavar='INT',
                        type=parserutils.type_positive_int,
                        help='')
    parser.add_argument(FLAG_Y_CELL_NUM,
                        metavar='INT',
                        type=parserutils.type_positive_int,
                        default=4,
                        help='')
    parser.add_argument(FLAG_Z_CELL_NUM,
                        metavar='INT',
                        type=parserutils.type_positive_int,
                        default=6,
                        help='')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


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
