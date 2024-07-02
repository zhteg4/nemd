import io
import re
import csv
import math
import scipy
import types
import base64
import itertools
import collections
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import constants

from nemd import oplsua
from nemd import symbols
from nemd import lammpsin
from nemd import structure
from nemd import numpyutils
from nemd import constants as nconstant

ID = 'id'
MOL_ID = 'mol_id'
TYPE_ID = 'type_id'
CHARGE = 'charge'
XU = symbols.XU
YU = symbols.YU
ZU = symbols.ZU
XYZU = symbols.XYZU
ATOM_COL = [ID, MOL_ID, TYPE_ID, CHARGE, XU, YU, ZU]

ATOM1 = 'atom1'
ATOM2 = 'atom2'
ATOM3 = 'atom3'
ATOM4 = 'atom4'

ENE = 'ene'
DIST = 'dist'


class Mass(pd.DataFrame):

    COLUMNS = 'columns'
    TO_CSV_KWARGS = dict(sep=' ',
                         header=False,
                         float_format='%.4f',
                         mode='a',
                         quotechar='#')
    NAME = 'Masses'
    COLUMN_LABELS = ['mass', 'comment']

    def __init__(self, data=None, index=None, columns=None, **kwargs):
        if not isinstance(data, pd.DataFrame) and columns is None:
            columns = self.COLUMN_LABELS
        super().__init__(data=data, index=index, columns=columns, **kwargs)
        if not isinstance(data, pd.DataFrame) and index is None:
            self.index = pd.RangeIndex(start=1, stop=self.shape[0] + 1)

    def to_csv(self, path_or_buf=None, as_block=True, **kwargs):
        if self.empty:
            return
        kwargs.update(self.TO_CSV_KWARGS)
        if not as_block:
            super().to_csv(path_or_buf=path_or_buf, **kwargs)
            return
        path_or_buf.write(self.NAME + '\n\n')
        super().to_csv(path_or_buf=path_or_buf, **kwargs)
        path_or_buf.write('\n')

    @classmethod
    @property
    def _constructor(cls):
        """
        Return the constructor of the class.

        :return 'Bond' class or subclass of 'Bond': the constructor of the class
        """
        return cls

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Vdw(Mass):

    NAME = 'Pair Coeffs'
    COLUMN_LABELS = [ENE, DIST]


class Bond(Mass):

    NAME = 'Bonds'
    DTYPE = 'dtype'
    DEFAULT_DTYPE = int
    ID_COLS = [ATOM1, ATOM2]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS

    def __init__(self, data=None, **kwargs):
        if data is None:
            dtype = kwargs.get(self.DTYPE, self.DEFAULT_DTYPE)
            data = {x: pd.Series(dtype=dtype) for x in self.COLUMN_LABELS}
        super().__init__(data=data, **kwargs)

    def append(self, *args, **kwargs):
        return self._append(*args, **kwargs)

    def mapIds(self, id_map):
        acopy = self.copy()
        acopy[self.ID_COLS] = id_map[acopy[self.ID_COLS]]
        return acopy

    def getPairs(self, step=1):
        slices = slice(None, None, step)
        return [tuple(sorted(x[slices])) for x in self[self.ID_COLS].values]

    def getFixed(self, func):
        vals = [x for x in self[TYPE_ID].unique() if func(x)]
        return pd.DataFrame({self.NAME: vals})

    @classmethod
    def concat(cls, objs, **kwargs):
        if not len(objs):
            return cls(None)
        data = pd.concat(objs, **kwargs)
        data.index = pd.RangeIndex(start=1, stop=data.shape[0] + 1)
        return data

    @classmethod
    def read_csv(cls, *args, **kwargs):
        kwargs.update(dict(names=cls.COLUMN_LABELS, sep=r'\s+'))
        return cls(pd.read_csv(*args, **kwargs))


class Angle(Bond):

    NAME = 'Angles'
    ID_COLS = [ATOM1, ATOM2, ATOM3]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS
    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['id_map']
    _internal_names_set = set(_internal_names)

    def getPairs(self, step=2):
        return super(Angle, self).getPairs(step=step)

    def setMap(self):
        atoms = self.drop(columns=[TYPE_ID])
        shape = 0 if self.empty else atoms.max().max() + 1
        if not shape:
            return
        self.id_map = np.zeros([shape] * 3, dtype=int)
        col1, col2, col3 = tuple(np.transpose(atoms.values))
        self.id_map[col1, col2, col3] = self.index
        self.id_map[col3, col2, col1] = self.index

    def min(self, x, key=None):
        indexes = self.id_map[tuple(np.transpose(x))]
        return min(indexes, key=lambda x: key(self.loc[x][TYPE_ID]))


class Dihedral(Bond):

    NAME = 'Dihedrals'
    ID_COLS = [ATOM1, ATOM2, ATOM3, ATOM4]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS

    def getPairs(self, step=3):
        return super(Dihedral, self).getPairs(step=step)


class Improper(Dihedral):

    NAME = 'Impropers'

    def getPairs(self):
        ids = [itertools.combinations(x, 2) for x in self[self.ID_COLS].values]
        return [tuple(sorted(y)) for x in ids for y in x]


class Conformer(structure.Conformer):

    @property
    def atoms(self):
        """
        Return atom information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as atom global
            ids, molecule ids, atom type ids, charges, coordinates.
        """
        atoms = self.GetOwningMol().atoms
        atoms.insert(0, MOL_ID, self.gid)
        for dim, vals in zip(symbols.XYZU, self.GetPositions().transpose()):
            atoms[dim] = vals
        atoms.index = self.id_map[atoms.index]
        return atoms

    @property
    def bonds(self):
        """
        Return bond information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as bond ids and
            bonded atom ids.
        """
        return self.GetOwningMol().bonds.mapIds(self.id_map)

    @property
    def angles(self):
        """
        Return angle information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as angle ids and
            connected atom ids.
        """
        return self.GetOwningMol().angles.mapIds(self.id_map)

    @property
    def dihedrals(self):
        """
        Return dihedral angle information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as dihedral ids
            and connected atom ids.
        """
        return self.GetOwningMol().dihedrals.mapIds(self.id_map)

    @property
    def impropers(self):
        """
        Return improper angle information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as improper ids
            and connected atom ids.
        """
        return self.GetOwningMol().impropers.mapIds(self.id_map)


class Mol(structure.Mol):

    ConfClass = Conformer
    RES_NUM = oplsua.RES_NUM
    IMPROPER_CENTER_SYMBOLS = symbols.CARBON + symbols.HYDROGEN
    INDEXES = [TYPE_ID, ATOM1, ATOM2]
    BONDS_KWARGS = dict(index=[TYPE_ID, ATOM1, ATOM2], dtype=int)

    def __init__(self, *args, ff=None, **kwargs):
        """
        :param ff 'Parser': the force field class.
        """
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.bonds = Bond()
        self.angles = Angle()
        self.dihedrals = Dihedral()
        self.impropers = Improper()
        self.symbol_impropers = {}
        self.nbr_charge = collections.defaultdict(float)
        if self.ff is None and self.struct and hasattr(self.struct, 'ff'):
            self.ff = self.struct.ff
        if self.ff is None:
            self.ff = oplsua.get_parser()
        if self.delay:
            return
        self.setTopo()

    def setTopo(self):
        """
        Set charge, bond, angle, dihedral, improper, and other topology params.
        """
        self.typeAtoms()
        self.balanceCharge()
        self.setBonds()
        self.setAngles()
        self.setDihedrals()
        self.setImpropers()
        self.removeAngles()

    def typeAtoms(self):
        """
        Assign atom types and other force field parameters.
        """
        self.ff.type(self)

    def balanceCharge(self):
        """
        Balance the charge when residues are not neutral.
        """
        # residual num: residual charge
        res_charge = collections.defaultdict(float)
        for atom in self.GetAtoms():
            res_num = atom.GetIntProp(self.RES_NUM)
            type_id = atom.GetIntProp(TYPE_ID)
            res_charge[res_num] += self.ff.charges[type_id]

        res_snacharge = {x: 0 for x, y in res_charge.items() if y}
        res_atom = {}
        for bond in self.GetBonds():
            batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
            bres_num = batom.GetIntProp(self.RES_NUM)
            eres_num = eatom.GetIntProp(self.RES_NUM)
            if bres_num == eres_num:
                continue
            # Bonded atoms in different residuals
            for atom, natom in [[batom, eatom], [eatom, batom]]:
                nres_num = natom.GetIntProp(self.RES_NUM)
                ncharge = res_charge[nres_num]
                if not ncharge:
                    continue
                # The natom lives in nres with total charge
                snatom_charge = abs(self.ff.charges[natom.GetIntProp(TYPE_ID)])
                if snatom_charge > res_snacharge[nres_num]:
                    res_atom[nres_num] = atom.GetIdx()
                    res_snacharge[nres_num] = snatom_charge

        for res, idx in res_atom.items():
            self.nbr_charge[idx] -= res_charge[res]

    @property
    def atoms(self):
        type_ids = [x.GetIntProp(TYPE_ID) for x in self.GetAtoms()]
        fchrg = [self.ff.charges[x] for x in type_ids]
        index = pd.Index([x.GetIdx() for x in self.GetAtoms()], name=ID)
        nchrg = [self.nbr_charge[x] for x in index]
        chrg = np.array([sum(x) for x in zip(fchrg, nchrg)])
        return pd.DataFrame({TYPE_ID: type_ids, CHARGE: chrg}, index=index)

    def setBonds(self):
        """
        Set bonding information.
        """
        bonds = [x for x in self.GetBonds()]
        data = {TYPE_ID: [self.ff.getMatchedBonds(x)[0].id for x in bonds]}
        data.update({ATOM1: [x.GetBeginAtom().GetIdx() for x in bonds]})
        data.update({ATOM2: [x.GetEndAtom().GetIdx() for x in bonds]})
        self.bonds = Bond(data)

    def setAngles(self):
        """
        Set angle force field matches.
        """
        angles = [y for x in self.GetAtoms() for y in self.ff.getAngleAtoms(x)]
        data = {TYPE_ID: [self.ff.getMatchedAngles(x)[0].id for x in angles]}
        idxs = [[y.GetIdx() for y in x] for x in zip(*angles)]
        data.update({x: y for x, y in zip(Angle.ID_COLS, idxs)})
        self.angles = Angle(data)

        self.angles.setMap()

    def setDihedrals(self):
        """
        Set the dihedral angles of the molecules.
        """
        dihes = [x for x in self.getDihAtoms()]
        data = {TYPE_ID: [self.ff.getMatchedDihedrals(x)[0].id for x in dihes]}
        idxs = [[y.GetIdx() for y in x] for x in zip(*dihes)]
        data.update({x: y for x, y in zip(Dihedral.ID_COLS, idxs)})
        self.dihedrals = Dihedral(data)

    def setImpropers(self):
        data = self.getImpropers()
        self.impropers = Improper(data)

    def getDihAtoms(self):
        """
        Get the dihedral atoms of this molecule.

        NOTE: Flipping the order the four dihedral atoms yields the same dihedral,
        and only one of them is returned.

        :return list of list: each sublist has four atom ids forming a dihedral angle.
        """
        atomss = [y for x in self.GetAtoms() for y in self.getDihedralAtoms(x)]
        # 1-2-3-4 and 4-3-2-1 are the same dihedral
        atomss_no_flip = []
        atom_idss = set()
        for atoms in atomss:
            atom_ids = tuple(x.GetIdx() for x in atoms)
            if atom_ids in atom_idss:
                continue
            atom_idss.add(atom_ids)
            atom_idss.add(atom_ids[::-1])
            atomss_no_flip.append(atoms)
        return atomss_no_flip

    def getDihedralAtoms(self, atom):
        """
        Get the dihedral atoms whose torsion bonded atoms contain this atom.

        :param atom 'rdkit.Chem.rdchem.Atom': the middle atom of the dihedral
        :return list of list: each sublist has four atom ids forming a dihedral
            angle.
        """
        dihe_atoms = []
        atomss = self.ff.getAngleAtoms(atom)
        atomss += [x[::-1] for x in atomss]
        for satom, matom, eatom in atomss:
            presented = set([matom.GetIdx(), eatom.GetIdx()])
            dihe_4ths = [
                y for x in self.ff.getAngleAtoms(eatom) for y in x
                if y.GetIdx() not in presented
            ]
            for dihe_4th in dihe_4ths:
                dihe_atoms.append([satom, matom, eatom, dihe_4th])

        return dihe_atoms

    def getImpropers(self, csymbols=IMPROPER_CENTER_SYMBOLS):
        """
        Set improper angles based on center atoms and neighbor symbols.

        :param csymbols str: each Char is one possible center element

        In short:
        1) sp2 sites and united atom CH groups (sp3 carbons) needs improper
         (though I saw a reference using improper for sp3 N)
        2) No rules for a center atom. (Charmm asks order for symmetricity)
        3) Number of internal geometry variables (3N_atom – 6) deletes one angle

        The details are the following:

        When the Weiner et al. (1984,1986) force field was developed, improper
        torsions were designated for specific sp2 sites, as well as for united
        atom CH groups - sp3 carbons with one implicit hydrogen.
        Ref: http://ambermd.org/Questions/improp.html

        There are no rules for a center atom. You simply define two planes, each
        defined by three atoms. The angle is given by the angle between these
        two planes. (from hess)
        ref: https://gromacs.bioexcel.eu/t/the-atom-order-i-j-k-l-in-defining-an
        -improper-dihedral-in-gromacs-using-the-opls-aa-force-field/3658

        The CHARMM convention in the definition of improper torsion angles is to
        list the central atom in the first position, while no rule exists for how
        to order the other three atoms.
        ref: Symmetrization of the AMBER and CHARMM Force Fields, J. Comput. Chem.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multipole and Gaussian
        Multipole Models
        """
        impropers = []
        for atom in self.GetAtoms():
            atom_symbol, neighbors = atom.GetSymbol(), atom.GetNeighbors()
            if atom_symbol not in csymbols or len(neighbors) != 3:
                continue
            if atom.GetSymbol() == symbols.NITROGEN and atom.GetHybridization(
            ) == Chem.rdchem.HybridizationType.SP3:
                continue
            # Sp2 carbon for planar, Sp3 with one H (CHR1R2R3) for chirality,
            # Sp2 N in Amino Acid
            neighbor_symbols = [x.GetSymbol() for x in neighbors]
            counted = self.ff.countSymbols(
                [str(oplsua.Parser.getAtomConnt(atom)), atom_symbol] +
                neighbor_symbols)
            improper_type_id = self.ff.improper_symbols[counted][0]
            # FIXME: see docstring for current investigation. (NO ACTIONS TAKEN)
            #  1) LAMMPS recommends the first to be the center, while the prm
            #  and literature order the third as the center.
            #  2) In addition, since improper has one non-connected edge,
            #  are the two non-edge atom selections important?
            #  3) Moreover, do we have to delete over constrained angle? If so,
            #  how about the one facing the non-connected edge?
            # My recommendation (not current implementation):
            # first plane: center + the two most heavy atom
            # second plane: the three non-center atoms
            # benefit: 1) O-C-O / O.O.R imposes symmetricity (RCOO)
            # 2) R-N-C / O.O.H exposes hydrogen out of plane vibration (RCNH)

            # My Implementation:
            # Use the center as the third according to "A New Force Field for
            # Molecular Mechanical Simulation of Nucleic Acids and Proteins"
            # No special treatment to the order of other atoms.

            # My Reasoning: first or third functions the same for planar
            # scenario as both 0 deg and 180 deg implies in plane. However,
            # center as first or third defines different planes, leading to
            # eiter ~45 deg or 120 deg as the equilibrium improper angle.
            # 120 deg sounds more plausible and thus the third is chosen to be
            # the center.
            atoms = [neighbors[0], neighbors[1], atom, neighbors[2]]
            improper = (improper_type_id, ) + tuple(x.GetIdx() for x in atoms)
            impropers.append(improper)
        return impropers

    def printImpropers(self):
        """
        Print all the possible improper angles in the force field file.
        """
        for symb, improper_ids in self.symbol_impropers.items():
            print(f"{symb} {self.ff.impropers[improper_ids[0]]}")
            impropers = [self.ff.impropers[x] for x in improper_ids]
            for improper in impropers:
                ids = [improper.id1, improper.id2, improper.id3, improper.id4]
                print(f"{[self.ff.atoms[x].description for x in ids]}")

    def removeAngles(self):
        """
        One improper adds one restraint and thus one angle is removed.

        e.g. NH3 if all three H-N-H angles are defined, you cannot control out
        of plane mode.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multi-pole and Gaussian
        Multi-xpole Models
        """
        columns = [ATOM2, ATOM1, ATOM4]
        cols = [[x, ATOM3, y] for x, y in itertools.combinations(columns, 2)]
        angles = zip(*[self.impropers[x].values for x in cols])
        index = [
            self.angles.min(x, key=lambda x: self.ff.angles[x].ene)
            for x in angles
        ]
        self.angles = self.angles.drop(index=index)

    def getFixed(self):
        """
        The lengths or angle values of these geometries remain unchanged during
        simulation.
        """
        bnd_types = self.bonds.getFixed(lambda x: self.ff.bonds[x].has_h)
        ang_types = self.angles.getFixed(lambda x: self.ff.angles[x].has_h)
        return bnd_types, ang_types

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight


class Base(lammpsin.In):

    LAMMPS_DESCRIPTION = 'LAMMPS Description # %s'

    ATOMS = 'atoms'
    BONDS = 'bonds'
    ANGLES = 'angles'
    DIHEDRALS = 'dihedrals'
    IMPROPERS = 'impropers'
    TOPO_CT = [ATOMS, BONDS, ANGLES, DIHEDRALS, IMPROPERS]

    ATOM_TYPES = 'atom types'
    BOND_TYPES = 'bond types'
    ANGLE_TYPES = 'angle types'
    DIHE_TYPES = 'dihedral types'
    IMPROP_TYPES = 'improper types'
    TYPE_CT = [ATOM_TYPES, BOND_TYPES, ANGLE_TYPES, DIHE_TYPES, IMPROP_TYPES]
    ALL_CT = TOPO_CT + TYPE_CT

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    BUFFER = [4., 4., 4.]

    MASSES = 'Masses'
    PAIR_COEFFS = 'Pair Coeffs'
    BOND_COEFFS = 'Bond Coeffs'
    ANGLE_COEFFS = 'Angle Coeffs'
    DIHEDRAL_COEFFS = 'Dihedral Coeffs'
    IMPROPER_COEFFS = 'Improper Coeffs'
    TYPE_BLK = [
        MASSES, PAIR_COEFFS, BOND_COEFFS, ANGLE_COEFFS, DIHEDRAL_COEFFS,
        IMPROPER_COEFFS
    ]
    BLK_COUNT = {x: y for x, y in zip(TYPE_BLK, TYPE_CT)}

    ATOMS_CAP = ATOMS.capitalize()
    BONDS_CAP = BONDS.capitalize()
    ANGLES_CAP = ANGLES.capitalize()
    DIHEDRALS_CAP = DIHEDRALS.capitalize()
    IMPROPERS_CAP = IMPROPERS.capitalize()
    TOPO_BLK = [ATOMS_CAP, BONDS_CAP, ANGLES_CAP, DIHEDRALS_CAP, IMPROPERS_CAP]
    BLK_COUNT.update({x: y for x, y in zip(TOPO_BLK, TOPO_CT)})
    BLK_MARKERS = TYPE_BLK + TOPO_BLK

    TOPO_TYPE = {
        BONDS_CAP: Bond,
        ANGLES_CAP: Angle,
        DIHEDRALS_CAP: Dihedral,
        IMPROPERS_CAP: Improper
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radii = None

    def setVdwRadius(self):
        """
        Set the vdw radius.
        """
        self.radii = Radius(self.vdws[DIST], self.atoms[TYPE_ID])

    def setClashExclusion(self, include14=True):
        """
        Bonded atoms and atoms in angles are in the exclusion. If include14=True,
        the dihedral angles are in the exclusion as well.

        :param include14 bool: If True, 1-4 interaction in a dihedral angle count
            as exclusion.
        """
        pairs = set(self.bonds.getPairs())
        pairs = pairs.union(self.angles.getPairs())
        pairs = pairs.union(self.impropers.getPairs())
        if include14:
            pairs = pairs.union(self.dihedrals.getPairs())
        for id1, id2 in pairs:
            self.excluded[id1].add(id2)
            self.excluded[id2].add(id1)


class Struct(structure.Struct, Base):

    MolClass = Mol
    TO_CSV_KWARGS = dict(sep=' ',
                         header=False,
                         float_format='%.4f',
                         mode='a',
                         quoting=csv.QUOTE_NONE)

    def __init__(self, struct=None, ff=None, options=None, **kwargs):
        """
        :param struct Struct: struct object with moelcules and conformers.
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(struct=struct, **kwargs)
        Base.__init__(self, options=options, **kwargs)
        self.ff = ff
        self.total_charge = 0.
        self.atm_types = numpyutils.IntArray()
        self.bnd_types = numpyutils.IntArray()
        self.ang_types = numpyutils.IntArray()
        self.dihe_types = numpyutils.IntArray()
        self.impr_types = numpyutils.IntArray()
        self.hdl = None
        self.warnings = []
        self.excluded = collections.defaultdict(set)
        self.initTypeMap()

    def initTypeMap(self):
        self.atm_types = numpyutils.IntArray(max(self.ff.atoms))
        self.bnd_types = numpyutils.IntArray(max(self.ff.bonds))
        self.ang_types = numpyutils.IntArray(max(self.ff.angles))
        self.dihe_types = numpyutils.IntArray(max(self.ff.dihedrals))
        self.impr_types = numpyutils.IntArray(max(self.ff.impropers))

    def addMol(self, mol):
        mol = super().addMol(mol)
        self.setTypeMap(mol)
        return mol

    def setTypeMap(self, mol):
        """
        Set the type map for atoms, bonds, angles, dihedrals, and impropers.
        """

        atypes = [x.GetIntProp(TYPE_ID) for x in mol.GetAtoms()]
        self.atm_types.add(atypes)
        self.bnd_types.add(mol.bonds[TYPE_ID])
        self.ang_types.add(mol.angles[TYPE_ID])
        self.dihe_types.add(mol.dihedrals[TYPE_ID])
        self.impr_types.add(mol.impropers[TYPE_ID])

    def writeData(self, nofile=False):
        """
        Write out LAMMPS data file.

        :param nofile bool: return the string instead of writing to a file if True
        """

        with io.StringIO() if nofile else open(self.datafile, 'w') as self.hdl:
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writePairCoeffs()
            self.writeBondCoeffs()
            self.writeAngleCoeffs()
            self.writeDiheCoeffs()
            self.writeImpropCoeffs()
            self.writeAtoms()
            self.writeBonds()
            self.writeAngles()
            self.writeDihedrals()
            self.writeImpropers()
            return self.getContents() if nofile else None

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom,
        bond, angle etc.
        """
        lmp_dsp = self.LAMMPS_DESCRIPTION % self.atom_style
        self.hdl.write(f"{lmp_dsp}\n\n")
        self.hdl.write(f"{self.atoms.shape[0]} {self.ATOMS}\n")
        self.hdl.write(f"{self.bonds.shape[0]} {self.BONDS}\n")
        self.hdl.write(f"{self.angles.shape[0]} {self.ANGLES}\n")
        self.hdl.write(f"{self.dihedrals.shape[0]} {self.DIHEDRALS}\n")
        self.hdl.write(f"{self.impropers.shape[0]} {self.IMPROPERS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        self.hdl.write(f"{self.atm_types.max()} {self.ATOM_TYPES}\n")
        self.hdl.write(f"{self.bnd_types.max()} {self.BOND_TYPES}\n")
        self.hdl.write(f"{self.ang_types.max()} {self.ANGLE_TYPES}\n")
        self.hdl.write(f"{self.dihe_types.max()} {self.DIHE_TYPES}\n")
        self.hdl.write(f"{self.impr_types.max()} {self.IMPROP_TYPES}\n")
        self.hdl.write("\n")

    def writeBox(self, min_box=None, buffer=None):
        """
        Write box information.

        :param min_box list: minimum box size
        :param buffer list: buffer in three dimensions
        """
        xyzs = self.getPositions()
        ctr = xyzs.mean(axis=0)
        box_hf = self.getHalfBox(xyzs, min_box=min_box, buffer=buffer)
        box = [[x - y, x + y, z] for x, y, z in zip(ctr, box_hf, self.LO_HI)]
        if self.box is not None:
            boxes = zip(box, np.array(self.box).reshape(-1, 2))
            box = [[*x, symbols.POUND, *y] for x, y in boxes]
        for line in box:
            line = [f'{x:.2f}' if isinstance(x, float) else x for x in line]
            self.hdl.write(f"{' '.join(line)}\n")
        self.hdl.write("\n")
        if self.density is None:
            return
        # Calculate density as the revised box may alter the box size.
        vol = math.prod([x * 2 * nconstant.ANG_TO_CM for x in box_hf])
        density = self.molecular_weight / vol / scipy.constants.Avogadro
        if np.isclose(self.density, density):
            return
        msg = f'The density of the final data file is {density:.4g} kg/cm^3'
        self.warnings.append(msg)

    def getHalfBox(self, xyzs, min_box=None, buffer=None):
        """
        Get the half box size based on interaction minimum, buffer, and structure
        span.

        :param xyzs 'numpy.ndarray': the xyz of the structure
        :param min_box list: minimum box size
        :param buffer list: the buffer in xyz dimensions (good for non-pbc)
        :return list of three floats: the xyz box limits.
        """
        if min_box is None:
            min_box = [1, 1, 1]
        if buffer is None:
            buffer = self.BUFFER  # yapf: disable
        box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
        if self.box is not None:
            box = [(x - y) for x, y in zip(self.box[1::2], self.box[::2])]
        box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]
        cut_off = min([self.options.lj_cut, self.options.coul_cut])
        if min(box_hf) < cut_off:
            # One particle interacts with another particle within cutoff twice:
            # within the box and across the PBC
            msg = f"The half box size ({min(box_hf):.2f} {symbols.ANGSTROM}) " \
                  f"is smaller than the {cut_off} {symbols.ANGSTROM} cutoff."
            self.warnings.append(msg)

        if self.conformer_total != 1:
            return box_hf
        # All-trans single molecule with internal tension runs into clashes
        # across PBCs and thus larger box is used.
        return [x * 1.2 for x in box_hf]

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.masses.to_csv(self.hdl)

    def writePairCoeffs(self):
        """
        Write pair coefficients.
        """
        self.vdws.to_csv(self.hdl)

    def writeBondCoeffs(self):
        """
        Write bond coefficients.
        """

        if not self.bnd_types.any():
            return

        self.hdl.write(f"{self.BOND_COEFFS}\n\n")
        for idx in self.bnd_types.on:
            bond = self.ff.bonds[idx]
            self.hdl.write(f"{self.bnd_types[idx]} {bond.ene} {bond.dist}\n")
        self.hdl.write("\n")

    def writeAngleCoeffs(self):
        """
        Write angle coefficients.
        """
        if not self.ang_types.any():
            return

        self.hdl.write(f"{self.ANGLE_COEFFS}\n\n")
        for idx in self.ang_types.on:
            ang = self.ff.angles[idx]
            self.hdl.write(f"{self.ang_types[idx]} {ang.ene} {ang.angle}\n")
        self.hdl.write("\n")

    def writeDiheCoeffs(self):
        """
        Write dihedral coefficients.
        """
        if not self.dihe_types.any():
            return

        self.hdl.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        for idx in self.dihe_types.on:
            params = [0., 0., 0., 0.]
            # LAMMPS: K1, K2, K3, K4 in 0.5*K1[1+cos(x)] + 0.5*K2[1-cos(2x)]...
            # OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            for ene_ang_n in self.ff.dihedrals[idx].constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if not params[ene_ang_n.n_parm]:
                    continue
                if (ene_ang_n.angle == 180.) ^ (not ene_ang_n.n_parm % 2):
                    params[ene_ang_n.n_parm] *= -1
            self.hdl.write(
                f"{self.dihe_types[idx]}  {' '.join(map(str, params))}\n")
        self.hdl.write("\n")

    def writeImpropCoeffs(self):
        """
        Write improper coefficients.
        """
        if not self.impr_types.any():
            return

        self.hdl.write(f"{self.IMPROPER_COEFFS}\n\n")
        for idx in self.impr_types.on:
            impr = self.ff.impropers[idx]
            # LAMMPS: K in K[1+d*cos(nx)] vs OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            sign = 1 if impr.angle == 0. else -1
            self.hdl.write(
                f"{self.impr_types[idx]} {impr.ene} {sign} {impr.n_parm}\n")
        self.hdl.write("\n")

    def writeAtoms(self):
        """
        Write atom coefficients.
        """
        self.hdl.write(f"{self.ATOMS.capitalize()}\n\n")
        self.atoms.to_csv(self.hdl, **self.TO_CSV_KWARGS)
        self.hdl.write(f"\n")
        if not round(self.atoms[CHARGE].sum(), 4):
            return
        msg = f'The system has a net charge of {self.atoms[CHARGE].sum():.4f}'
        self.warnings.append(msg)

    def writeBonds(self):
        """
        Write bond coefficients.
        """
        self.bonds.to_csv(self.hdl)

    def writeAngles(self):
        """
        Write angle coefficients.
        """
        self.angles.to_csv(self.hdl)

    def writeDihedrals(self):
        """
        Write dihedral coefficients.
        """
        self.dihedrals.to_csv(self.hdl)

    def writeImpropers(self):
        """
        Write improper coefficients.
        """
        self.impropers.to_csv(self.hdl)

    def getContents(self):
        """
        Return datafile contents in base64 encoding.

        :return `bytes`: the contents of the data file in base64 encoding.
        """
        self.hdl.seek(0)
        contents = base64.b64encode(self.hdl.read().encode("utf-8"))
        return b','.join([b'lammps_datafile', contents])

    def writeRun(self, *arg, **kwarg):
        """
        Write command to further equilibrate the system with molecules
        information considered.
        """
        btypes, atypes = self.getFixed()
        testing = self.conformer_total == 1 and self.atom_total < 100
        struct_info = types.SimpleNamespace(btypes=btypes,
                                            atypes=atypes,
                                            testing=testing)
        super().writeRun(*arg, struct_info=struct_info, **kwarg)

    def getFixed(self):
        data = [x.getFixed() for x in self.molecules]
        bonds, angles = list(map(list, zip(*data)))
        bonds = Bond.concat([x for x in bonds if not x.empty])
        angles = Angle.concat([x for x in angles if not x.empty])
        bond_types = self.bnd_types[bonds].flatten()
        angle_types = self.ang_types[angles].flatten()
        return [' '.join(map(str, x)) for x in [bond_types, angle_types]]

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        return np.isclose(self.atoms[CHARGE], 0, 0.001).any()

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return sum([x.mw * x.GetNumConformers() for x in self.molecules])

    mw = molecular_weight

    @property
    def atoms(self):
        data = pd.concat(x.atoms for x in self.conformer)
        data[TYPE_ID] = self.atm_types[data[TYPE_ID]]
        return data

    @property
    def bonds(self):
        bonds = [x.bonds for x in self.conformer if not x.bonds.empty]
        bonds = Bond.concat(bonds, axis=0)
        bonds[TYPE_ID] = self.bnd_types[bonds[TYPE_ID]]
        return bonds

    @property
    def angles(self):
        angles = [x.angles for x in self.conformer if not x.angles.empty]
        angles = Angle.concat(angles, axis=0)
        angles[TYPE_ID] = self.ang_types[angles[TYPE_ID]]
        return angles

    @property
    def dihedrals(self):
        dihes = [x.dihedrals for x in self.conformer if not x.dihedrals.empty]
        dihes = Dihedral.concat(dihes, axis=0)
        dihes[TYPE_ID] = self.dihe_types[dihes[TYPE_ID]]
        return dihes

    @property
    def impropers(self):
        imprps = [x.impropers for x in self.conformer if not x.impropers.empty]
        imprps = Improper.concat(imprps, axis=0)
        imprps[TYPE_ID] = self.impr_types[imprps[TYPE_ID]]
        return imprps

    @property
    def masses(self):
        masses = [self.ff.atoms[x] for x in self.atm_types.on]
        masses = Mass([[x.mass, f" {x.description} {x.symbol} {x.id} "]
                       for x in masses])
        return masses

    @property
    def vdws(self):
        vdws = [self.ff.vdws[x] for x in self.atm_types.on]
        vdws = Vdw([[x.ene, x.dist] for x in vdws])
        return vdws


class DataFileReader(Base):
    """
    LAMMPS Data file reader
    """

    MASS = 'mass'
    ELE = 'ele'
    MASS_COL = [ID, MASS, ELE]
    BLK_RE = re.compile(f"^{'|'.join(Base.BLK_MARKERS)}$")
    CT_RE = re.compile(f"^[0-9]+\s+({'|'.join(Base.ALL_CT)})$")
    FLT_RE = "[+-]?[\d\.\d]+"
    BOX_RE = re.compile(f"^{FLT_RE}\s+{FLT_RE}\s+({'|'.join(Base.LO_HI)}).*$")

    def __init__(self, data_file=None, contents=None):
        """
        :param data_file str: data file with path
        :param contents `bytes`: parse the contents if data_file not provided.
        """
        self.data_file = data_file
        self.contents = contents
        self.lines = None
        self.masses = {}
        self.atoms = {}
        self.vdws = {}
        self.excluded = collections.defaultdict(set)
        self.blk_idx = {}
        self.count = {x: 0 for x in self.ALL_CT}
        self.box = {}

    def run(self):
        """
        Main method to read and parse the data file.
        """
        self.setLines()
        self.indexLines()
        self.setDescription()
        self.setMasses()
        self.setPairCoeffs()
        self.setAtoms()

    def setLines(self):
        """
        Read the data file or content into lines.
        """
        if self.data_file:
            with open(self.data_file, 'r') as df_fh:
                self.lines = df_fh.readlines()
                return

        content_type, content_string = self.contents.split(b',')
        decoded = base64.b64decode(content_string)
        self.lines = decoded.decode("utf-8").splitlines()

    def indexLines(self):
        """
        Index the lines by block markers.
        """
        for idx, line in enumerate(self.lines):
            match = self.BLK_RE.match(line)
            if not match:
                continue
            self.blk_idx[match.group()] = idx

    def setDescription(self):
        """
        Parse the description section for topo counts, type counts, and box size
        """
        for line in self.lines[:min(self.blk_idx.values())]:
            match = self.CT_RE.match(line)
            if match:
                # 'atoms': 1620, 'bonds': 1593, 'angles': 1566 ...
                # 'atom types': 7, 'bond types': 6, 'angle types': 5 ...
                self.count[match.group(1)] = int(line.split(match.group(1))[0])
                continue
            match = self.BOX_RE.match(line)
            if match:
                # 'xlo xhi': [-7.12, 35.44], 'ylo yhi': [-7.53, 34.26], ..
                val = [float(x) for x in line.split(match.group(1))[0].split()]
                self.box[match.group(1)] = val
                continue

    def setMasses(self):
        """
        Parse the mass section for masses and elements.
        """
        sidx = self.blk_idx[self.MASSES] + 2
        data = []
        for idx in range(sidx, sidx + self.count[self.ATOM_TYPES]):
            splitted = self.lines[idx].split()
            data.append([splitted[0], splitted[1], splitted[-2]])
        self.masses = pd.DataFrame(data, columns=self.MASS_COL)
        self.masses[ID] = self.masses[ID].astype(int)
        self.masses[self.MASS] = self.masses[self.MASS].astype(float)

    def setPairCoeffs(self):
        """
        Paser the pair coefficient section.
        """

        if self.PAIR_COEFFS not in self.blk_idx:
            return
        sidx = self.blk_idx[self.PAIR_COEFFS] + 2
        lines = self.lines[sidx:sidx + self.count[self.ATOM_TYPES]]
        data = pd.read_csv(io.StringIO(''.join(lines)),
                           names=[ID] + Vdw.COLUMN_LABELS,
                           sep=r'\s+')
        self.vdws = data.set_index(ID)

    def read(self, name):
        if name not in self.blk_idx:
            return self.TOPO_TYPE[name].read_csv(io.StringIO(''))
        sidx = self.blk_idx[name] + 2
        lines = self.lines[sidx:sidx + self.count[self.BLK_COUNT[name]]]
        io_str = io.StringIO(''.join(lines))
        return self.TOPO_TYPE[name].read_csv(io_str)

    def getBox(self):
        """
        Get the box.

        :return list of float: xlo, xhi, ylo, yhi, zlo, zhi
        """
        return [y for x in self.box_dsp.values() for y in x]

    def setAtoms(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        sidx = self.blk_idx[self.ATOMS_CAP] + 2
        lines = self.lines[sidx:sidx + self.count[self.ATOMS]]
        data = pd.read_csv(io.StringIO(''.join(lines)),
                           names=ATOM_COL,
                           sep=r'\s+')
        self.atoms = data.set_index(ID)

    def gidFromEle(self, ele):
        if ele is None:
            return self.atoms.index.tolist()
        type_id = self.masses[ID][self.masses[self.ELE] == ele]
        return self.atoms.index[self.atoms[TYPE_ID] ==
                                type_id.iloc[0]].tolist()

    @property
    def bonds(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        return self.read(name=self.BONDS_CAP)

    @property
    def angles(self):
        """
        Parse the angle section for angle id and constructing atoms.
        """
        return self.read(name=self.ANGLES_CAP)

    @property
    def dihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.
        """
        return self.read(name=self.DIHEDRALS_CAP)

    @property
    def impropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.
        """
        return self.read(name=self.IMPROPERS_CAP)

    def setClashParams(self, include14=False):
        """
        Set clash check related parameters including pair radii and exclusion.

        :param include14 bool: whether to include atom separated by 2 bonds for
            clash check.
        """
        self.setClashExclusion(include14=not include14)
        self.setPairCoeffs()
        self.setVdwRadius()


class Radius(numpyutils.Array):
    """
    Class to get vdw radius from atom id pair.

    NOTE: the scaled radii here are more of diameters (or distance)
        between two sites.
    """

    MIN_DIST = 1.4
    SCALE = 0.45

    def __new__(cls, dists, atypes, *args, **kwargs):
        """
        :param dists pandas.Series: type id (index), atom radius (value)
        :param atypes pandas.Series: global atom id (index), atom type (value)
        """
        # Data.GEOMETRIC is optimized for speed and is supported
        kwargs = dict(index=range(dists.index.max() + 1), fill_value=0)
        radii = dists.reindex(**kwargs).values.tolist()
        radii = np.full((len(radii), len(radii)), radii, dtype='float16')
        radii *= radii.transpose()
        radii = np.sqrt(radii)
        radii *= pow(2, 1 / 6) * cls.SCALE
        radii[radii < cls.MIN_DIST] = cls.MIN_DIST
        obj = np.asarray(radii).view(cls)
        kwargs = dict(index=range(atypes.index.max() + 1), fill_value=0)
        obj.id_map = atypes.reindex(**kwargs).values
        return obj
