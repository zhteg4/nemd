import io
import re
import types
import base64
import itertools
import collections
import numpy as np
import pandas as pd
from rdkit import Chem

from nemd import oplsua
from nemd import symbols
from nemd import lammpsin
from nemd import structure
from nemd import numpyutils

ID = 'id'
TYPE_ID = 'type_id'

ATOM1 = 'atom1'
ATOM2 = 'atom2'
ATOM3 = 'atom3'
ATOM4 = 'atom4'


class Block(pd.DataFrame):

    NAME = 'Block'
    COLUMN_LABELS = ['column_labels']
    LABEL = 'label'
    QUOTECHAR = symbols.POUND

    def __init__(self, data=None, index=None, columns=None, **kwargs):
        """
        Initialize the Mass object.

        :param data: `pandas.DataFrame`: the data to initialize the object.
        :param index: `pandas.Index`: the index to initialize the object.
        :param columns: `list`: the column labels to initialize the object.
        """
        if not isinstance(data, pd.DataFrame) and columns is None:
            columns = self.COLUMN_LABELS
        super().__init__(data=data, index=index, columns=columns, **kwargs)

    @classmethod
    @property
    def _constructor(cls):
        """
        Return the constructor of the class.

        :return (sub)-class of 'Bond': the constructor of the class
        """
        return cls

    @classmethod
    def new(cls, *args, **kwargs):
        """
        Create a new instance of the class.
        """
        return cls(*args, **kwargs)

    @classmethod
    def read_csv(cls,
                 *args,
                 names=None,
                 index_col=None,
                 sep=r'\s+',
                 quotechar=QUOTECHAR,
                 **kwargs):
        if names is None:
            names = [ID] + cls.COLUMN_LABELS
            if index_col is None:
                index_col = ID
        df = pd.read_csv(*args,
                         names=names,
                         index_col=index_col,
                         sep=sep,
                         quotechar=quotechar,
                         **kwargs)
        return cls(df)

    def to_csv(self,
               path_or_buf=None,
               as_block=True,
               sep=' ',
               header=False,
               float_format='%.4f',
               mode='a',
               quotechar=QUOTECHAR,
               **kwargs):
        """
        Write the data to a file buffer.

        :param path_or_buf '_io.TextIOWrapper': the buffer to write to.
        :param as_block `bool`: whether to write the data as a block.
        :param sep `str`: the separator to use.
        :param header `bool`: whether to write the column names as the header.
        :param float_format `str`: the format to use for floating point numbers.
        :param mode `str`: the mode to use for writing.
        :param quotechar `str`: the quote character to use.
        """

        if self.empty:
            return

        content = self.NAME + '\n\n' if as_block and self.NAME else ''
        content += super().to_csv(sep=sep,
                                  header=header,
                                  float_format=float_format,
                                  mode=mode,
                                  quotechar=quotechar,
                                  **kwargs)
        if as_block:
            content += '\n'
        path_or_buf.write(content)


class Box(Block):

    NAME = ''
    LO, HI = 'lo', 'hi'
    COLUMN_LABELS = [LO, HI]
    INDEX = ['x', 'y', 'z']
    ORIGIN = [0, 0, 0]
    LIMIT_CMT = '{limit}_cmt'
    LO_LABEL, HI_LABEL = LIMIT_CMT.format(limit=LO), LIMIT_CMT.format(limit=HI)
    LO_CMT = [x + y for x, y in itertools.product(INDEX, [LO])]
    HI_CMT = [x + y for x, y in itertools.product(INDEX, [HI])]

    def __init__(self, data=None, index=INDEX, **kwargs):
        super().__init__(data=data, index=index, **kwargs)

    @classmethod
    def fromEdges(cls, edges):
        return cls(data={cls.LO: cls.ORIGIN, cls.HI: edges})

    @property
    def span(self):
        return self.hi - self.lo

    def to_csv(self, fh, index=False, **kwargs):
        self[self.LO_LABEL] = self.LO_CMT
        self[self.HI_LABEL] = self.HI_CMT
        super().to_csv(fh, index=index, **kwargs)
        self.drop(columns=[self.LO_LABEL, self.HI_LABEL], inplace=True)

    def getPoint(self):
        point = np.random.rand(3) * self.span
        return point + self.lo


class Mass(Block):
    """
    The masses of the atoms in the system.
    """

    NAME = 'Masses'
    COLUMN_LABELS = ['mass', 'comment']
    LABEL = 'atom types'

    def __init__(self, data=None, index=None, columns=None, **kwargs):
        """
        Initialize the Mass object.

        :param data: `pandas.DataFrame`: the data to initialize the object.
        :param index: `pandas.Index`: the index to initialize the object.
        :param columns: `list`: the column labels to initialize the object.
        """
        super().__init__(data=data, index=index, columns=columns, **kwargs)
        if not isinstance(data, pd.DataFrame) and index is None:
            self.index = pd.RangeIndex(start=1, stop=self.shape[0] + 1)

    def writeCount(self, fh):
        fh.write(f'{self.shape[0]} {self.LABEL}\n')


class PairCoeff(Mass):
    """
    The pair coefficients between non-bonded atoms in the system.
    """

    NAME = 'Pair Coeffs'
    ENE = 'ene'
    DIST = 'dist'
    COLUMN_LABELS = [ENE, DIST]
    LABEL = 'atom types'


class BondCoeff(PairCoeff):
    """
    The bond coefficients between bonded atoms in the system.
    """
    NAME = 'Bond Coeffs'
    LABEL = 'bond types'


class AngleCoeff(PairCoeff):
    """
    The bond coefficients between bonded atoms in the system.
    """
    NAME = 'Angle Coeffs'
    DEG = 'deg'
    COLUMN_LABELS = [PairCoeff.ENE, DEG]
    LABEL = 'angle types'


class DihedralCoeff(AngleCoeff):
    """
    The bond coefficients between bonded atoms in the system.
    """
    NAME = 'Dihedral Coeffs'
    COLUMN_LABELS = ['k1', 'k2', 'k3', 'k4']
    LABEL = 'dihedral types'


class ImproperCoeff(AngleCoeff):
    """
    The bond coefficients between bonded atoms in the system.
    """
    NAME = 'Improper Coeffs'
    COLUMN_LABELS = ['k', 'd', 'n']
    LABEL = 'improper types'


class Atom(Mass):

    NAME = 'Atoms'
    MOL_ID = 'mol_id'
    CHARGE = 'charge'
    XU = symbols.XU
    YU = symbols.YU
    ZU = symbols.ZU
    XYZU = symbols.XYZU
    COLUMN_LABELS = [MOL_ID, TYPE_ID, CHARGE, XU, YU, ZU]
    LABEL = 'atoms'

    def mapIds(self, id_map):
        acopy = self.copy()
        acopy.index = id_map[acopy.index]
        return acopy


class Bond(Atom):

    NAME = 'Bonds'
    ID_COLS = [ATOM1, ATOM2]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS
    DTYPE = 'dtype'
    DEFAULT_DTYPE = int
    LABEL = 'bonds'

    def __init__(self, data=None, **kwargs):
        if data is None:
            dtype = kwargs.get(self.DTYPE, self.DEFAULT_DTYPE)
            data = {x: pd.Series(dtype=dtype) for x in self.COLUMN_LABELS}
        super().__init__(data=data, **kwargs)

    def mapIds(self, id_map):
        acopy = self.copy()
        acopy[self.ID_COLS] = id_map[acopy[self.ID_COLS]]
        return acopy

    def getPairs(self, step=1):
        slices = slice(None, None, step)
        return [tuple(sorted(x[slices])) for x in self[self.ID_COLS].values]

    def getRigid(self, func):
        vals = [x for x in self[TYPE_ID].unique() if func(x)]
        return pd.DataFrame({self.NAME: vals})

    @classmethod
    def concat(cls, objs, **kwargs):
        if not len(objs):
            return cls(None)
        data = pd.concat(objs, **kwargs)
        data.index = pd.RangeIndex(start=1, stop=data.shape[0] + 1)
        return data


class Angle(Bond):

    NAME = 'Angles'
    ID_COLS = [ATOM1, ATOM2, ATOM3]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS
    LABEL = 'angles'
    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['id_map']
    _internal_names_set = set(_internal_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_map = None

    def getPairs(self, step=2):
        return super(Angle, self).getPairs(step=step)

    def select(self, atom_ids):
        """
        Get the angles indexes whose energy is the lowest.

        :param atom_ids `numpy.ndarray`: list of an angle atom id tuples
        """
        if self.id_map is None:
            shape = 0 if self.empty else self[self.ID_COLS].max().max() + 1
            self.id_map = np.zeros([shape] * len(self.ID_COLS), dtype=int)
            col1, col2, col3 = tuple(np.transpose(self[self.ID_COLS].values))
            self.id_map[col1, col2, col3] = self.index
            self.id_map[col3, col2, col1] = self.index

        return self.loc[self.id_map[tuple(np.transpose(atom_ids))]]

    def getIndex(self, func):
        """
        Get the index of the angle with the lowest energy.

        :param key func: a function to get the angle energy from the type.
        """
        return min(self.index, key=lambda x: func(self.loc[x].type_id))


class Dihedral(Bond):

    NAME = 'Dihedrals'
    ID_COLS = [ATOM1, ATOM2, ATOM3, ATOM4]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS
    LABEL = 'dihedrals'

    def getPairs(self, step=3):
        return super(Dihedral, self).getPairs(step=step)


class Improper(Dihedral):

    NAME = 'Impropers'
    LABEL = 'impropers'

    def getPairs(self):
        ids = [itertools.combinations(x, 2) for x in self[self.ID_COLS].values]
        return [tuple(sorted(y)) for x in ids for y in x]

    def getAngles(self):
        columns = [ATOM2, ATOM1, ATOM4]
        cols = [[x, ATOM3, y] for x, y in itertools.combinations(columns, 2)]
        return np.array([x for x in zip(*[self[x].values for x in cols])])


class Conformer(structure.Conformer):

    @property
    def atoms(self):
        """
        Return atom information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as atom global
            ids, molecule ids, atom type ids, charges, coordinates.
        """
        atoms = self.GetOwningMol().atoms.mapIds(self.id_map)
        atoms.mol_id = self.gid
        atoms[Atom.XYZU] = self.GetPositions()
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

    def __init__(self, *args, ff=None, **kwargs):
        """
        :param ff 'Parser': the force field class.
        """
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.atoms = Atom()
        self.bonds = Bond()
        self.angles = Angle()
        self.dihedrals = Dihedral()
        self.impropers = Improper()
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
        self.setAtoms()
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

    def setAtoms(self):
        type_ids = [x.GetIntProp(TYPE_ID) for x in self.GetAtoms()]
        fchrg = [self.ff.charges[x] for x in type_ids]
        aids = [x.GetIdx() for x in self.GetAtoms()]
        nchrg = [self.nbr_charge[x] for x in aids]
        chrg = [sum(x) for x in zip(fchrg, nchrg)]
        self.atoms = Atom({TYPE_ID: type_ids, Atom.CHARGE: chrg}, index=aids)

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
        imprps = [x for x in self.getImproperAtoms()]
        data = {TYPE_ID: [self.ff.getMatchedImpropers(x)[0] for x in imprps]}
        idxs = [[y.GetIdx() for y in x] for x in zip(*imprps)]
        data.update({x: y for x, y in zip(Improper.ID_COLS, idxs)})
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

    def getImproperAtoms(self):
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
        # FIXME: LAMMPS recommends the first to be the center, while the prm
        # and literature order the third as the center.
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

        atoms = []
        for atom in self.GetAtoms():
            if atom.GetTotalDegree() != 3:
                continue
            match atom.GetSymbol():
                case symbols.CARBON:
                    # Planar Sp2 carbonyl carbon (R-COOH)
                    # tetrahedral Sp3 carbon with one implicit H (CHR1R2R3)
                    atoms.append(atom)
                case symbols.NITROGEN:
                    if atom.GetHybridization(
                    ) == Chem.rdchem.HybridizationType.SP2:
                        # Sp2 N in Amino Acid or Dimethylformamide
                        atoms.append(atom)
        neighbors = [x.GetNeighbors() for x in atoms]
        return [[y[0], y[1], x, y[2]] for x, y in zip(atoms, neighbors)]

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
        angles = [self.angles.select(x) for x in self.impropers.getAngles()]
        index = [x.getIndex(lambda x: self.ff.angles[x].ene) for x in angles]
        self.angles = self.angles.drop(index=index)

    def getRigid(self):
        """
        The bond and angle are rigid during simulation.

        :return
        """
        bnd_types = self.bonds.getRigid(lambda x: self.ff.bonds[x].has_h)
        ang_types = self.angles.getRigid(lambda x: self.ff.angles[x].has_h)
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

    DESCR = 'LAMMPS Description # {style}'

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    BUFFER = [4., 4., 4.]

    TYPE_CLASSES = [
        Mass, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff, ImproperCoeff
    ]
    TOPO_CLASSES = [Atom, Bond, Angle, Dihedral, Improper]

    BLOCK_CLASSES = TYPE_CLASSES + TOPO_CLASSES
    BLOCK_NAMES = [x.NAME for x in BLOCK_CLASSES]
    BLOCK_LABELS = [x.LABEL for x in BLOCK_CLASSES]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radii = None
        self.excluded = collections.defaultdict(set)

    def setClashParams(self, include14=False):
        """
        Set clash check related parameters including pair radii and exclusion.

        :param include14 bool: whether to include atom separated by 2 bonds for
            clash check.
        """
        self.setVdwRadius()
        self.setClashExclusion(include14=not include14)

    def setVdwRadius(self):
        """
        Set the vdw radius.
        """
        self.radii = Radius(self.pair_coeffs.dist, self.atoms.type_id)

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
        self.warnings = []
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
            self.hdl.write(f"{self.DESCR.format(style=self.atom_style)}\n\n")
            # Topology counting
            self.atoms.writeCount(self.hdl)
            self.bonds.writeCount(self.hdl)
            self.angles.writeCount(self.hdl)
            self.dihedrals.writeCount(self.hdl)
            self.impropers.writeCount(self.hdl)
            self.hdl.write("\n")
            # Type counting
            self.masses.writeCount(self.hdl)
            self.bond_coeffs.writeCount(self.hdl)
            self.angle_coeffs.writeCount(self.hdl)
            self.dihedral_coeffs.writeCount(self.hdl)
            self.improper_coeffs.writeCount(self.hdl)
            self.hdl.write("\n")
            # Box boundary
            self.box.to_csv(self.hdl)
            # Interaction coefficients
            self.masses.to_csv(self.hdl)
            self.pair_coeffs.to_csv(self.hdl)
            self.bond_coeffs.to_csv(self.hdl)
            self.angle_coeffs.to_csv(self.hdl)
            self.dihedral_coeffs.to_csv(self.hdl)
            self.improper_coeffs.to_csv(self.hdl)
            # Topology details
            self.atoms.to_csv(self.hdl)
            self.bonds.to_csv(self.hdl)
            self.angles.to_csv(self.hdl)
            self.dihedrals.to_csv(self.hdl)
            self.impropers.to_csv(self.hdl)

            return self.getContents() if nofile else None

    def getContents(self):
        """
        Return datafile contents in base64 encoding.

        :return `bytes`: the contents of the data file in base64 encoding.
        """
        self.hdl.seek(0)
        contents = base64.b64encode(self.hdl.read().encode("utf-8"))
        return b','.join([b'lammps_datafile', contents])

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
    def pair_coeffs(self):
        vdws = [self.ff.vdws[x] for x in self.atm_types.on]
        return PairCoeff([[x.ene, x.dist] for x in vdws])

    @property
    def bond_coeffs(self):
        bonds = [self.ff.bonds[x] for x in self.bnd_types.on]
        return BondCoeff([[x.ene, x.dist] for x in bonds])

    @property
    def angle_coeffs(self):
        angles = [self.ff.angles[x] for x in self.ang_types.on]
        return AngleCoeff([[x.ene, x.deg] for x in angles])

    @property
    def dihedral_coeffs(self):

        def getParams(ene_ang_ns):
            params = [0., 0., 0., 0.]
            # LAMMPS: K1, K2, K3, K4 in 0.5*K1[1+cos(x)] + 0.5*K2[1-cos(2x)]...
            # OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            for ene_ang_n in ene_ang_ns:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if not params[ene_ang_n.n_parm]:
                    continue
                if (ene_ang_n.angle == 180.) ^ (not ene_ang_n.n_parm % 2):
                    params[ene_ang_n.n_parm] *= -1
            return params

        dihes = [self.ff.dihedrals[x] for x in self.dihe_types.on]
        return DihedralCoeff([getParams(x.constants) for x in dihes])

    @property
    def improper_coeffs(self):
        imprps = [self.ff.impropers[x] for x in self.impr_types.on]
        # LAMMPS: K in K[1+d*cos(nx)] vs OPLS: [1 + cos(nx-gama)]
        # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
        imprps = [[x.ene, 1 if x.deg == 0. else -1, x.n_parm] for x in imprps]
        return ImproperCoeff(imprps)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return sum([x.mw * x.GetNumConformers() for x in self.molecules])

    mw = molecular_weight

    def getHalfBox(self, min_box=None, buffer=None):
        """
        Get the half box size based on interaction minimum, buffer, and structure
        span.

        :param min_box list: minimum box size
        :param buffer list: the buffer in xyz dimensions (good for non-pbc)
        :return list of three floats: the xyz box limits.
        """
        if min_box is None:
            min_box = [1, 1, 1]
        if buffer is None:
            buffer = self.BUFFER
        xyzs = self.getPositions()
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

    def writeRun(self, *arg, **kwarg):
        """
        Write command to further equilibrate the system with molecules
        information considered.
        """
        btypes, atypes = self.getRigid()
        testing = self.conformer_total == 1 and self.atom_total < 100
        struct_info = types.SimpleNamespace(btypes=btypes,
                                            atypes=atypes,
                                            testing=testing)
        super().writeRun(*arg, struct_info=struct_info, **kwarg)

    def getRigid(self):
        data = [x.getRigid() for x in self.molecules]
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
        return np.isclose(self.atoms.charge, 0, 0.001).any()


class DataFileReader(Base):
    """
    LAMMPS Data file reader
    """

    NAME_RE = re.compile(f"^{'|'.join(Base.BLOCK_NAMES)}$")
    COUNT_RE = re.compile(f"^[0-9]+\s+({'|'.join(Base.BLOCK_LABELS)})$")
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
        self.blk_idx = {}
        self.count = {x: 0 for x in self.BLOCK_LABELS}
        self.box = {}

    def run(self):
        """
        Main method to read and parse the data file.
        """
        self.setLines()
        self.indexLines()
        self.setDescription()

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
            match = self.NAME_RE.match(line)
            if not match:
                continue
            self.blk_idx[match.group()] = idx

    def setDescription(self):
        """
        Parse the description section for topo counts, type counts, and box size
        """
        for line in self.lines[:min(self.blk_idx.values())]:
            match = self.COUNT_RE.match(line)
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

    def getBox(self):
        """
        Get the box.

        :return list of float: xlo, xhi, ylo, yhi, zlo, zhi
        """
        return [y for x in self.box_dsp.values() for y in x]

    def read(self, BlockClass):
        if BlockClass.NAME not in self.blk_idx:
            return BlockClass.read_csv(io.StringIO(''))
        sidx = self.blk_idx[BlockClass.NAME] + 2
        lines = self.lines[sidx:sidx + self.count[BlockClass.LABEL]]
        return BlockClass.read_csv(io.StringIO(''.join(lines)))

    def gidFromEle(self, ele):
        if ele is None:
            return self.atoms.index.tolist()

        type_id = [
            i for i, x in self.masses.comment.items() if x.split()[-2] == ele
        ][0]
        return self.atoms.index[self.atoms[TYPE_ID] == type_id].tolist()

    @property
    def masses(self):
        """
        Parse the mass section for masses and elements.
        """
        return self.read(Mass)

    @property
    def pair_coeffs(self):
        """
        Paser the pair coefficient section.
        """
        return self.read(PairCoeff)

    @property
    def atoms(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        return self.read(Atom)

    @property
    def bonds(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        return self.read(Bond)

    @property
    def angles(self):
        """
        Parse the angle section for angle id and constructing atoms.
        """
        return self.read(Angle)

    @property
    def dihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.
        """
        return self.read(Dihedral)

    @property
    def impropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.
        """
        return self.read(Improper)


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
