import io
import re
import types
import base64
import itertools
import functools
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
TYPE_ID = oplsua.TYPE_ID
ATOM1 = 'atom1'
ATOM2 = 'atom2'
ATOM3 = 'atom3'
ATOM4 = 'atom4'


class Block(pd.DataFrame):
    """
    Base class to handle a datafile block.
    """

    NAME = 'Block'
    COLUMN_LABELS = ['column_labels']
    ID_COLS = None
    TYPE_COL = None
    POUND = symbols.POUND
    SPACE = symbols.SPACE
    SPACE_PATTERN = symbols.SPACE_PATTERN
    LABEL = 'label'

    def __init__(self, data=None, index=None, columns=None, **kwargs):
        """
        Initialize the Mass object.

        :param data: `pandas.DataFrame`: the data to initialize the object.
        :param index: `pandas.Index`: the index to initialize the object.
        :param columns: `list`: the column labels to initialize the object.
        """
        if not isinstance(data, pd.DataFrame) and columns is None:
            columns = self.COLUMN_LABELS
        if isinstance(data, int):
            data = np.ones((data, len(columns)), dtype=int)
        super().__init__(data=data, index=index, columns=columns, **kwargs)

    @classmethod
    @property
    def _constructor(cls):
        """
        Return the constructor of the class.

        :return (sub-)class of 'Block': the constructor of the class
        """
        return cls

    @classmethod
    def new(cls, *args, **kwargs):
        """
        Create a new instance of the (sub-)class.

        :return instance of Block (sub-)class: the new instance.
        """
        return cls(*args, **kwargs)

    @classmethod
    def fromLines(cls,
                  lines,
                  *args,
                  names=None,
                  index_col=None,
                  header=None,
                  sep=SPACE_PATTERN,
                  quotechar=POUND,
                  **kwargs):
        """
        Construct a new instance from a list of lines.

        :param names list: Sequence of column labels to apply.
        :param index_col int: Column(s) to use as row label(s)
        :param header int, ‘infer’ or None: the row number defining the header
        :param lines list: list of lines to parse.
        :param sep str: Character or regex pattern to treat as the delimiter.
        :param quotechar str: Character used to denote the start and end of a
            quoted item

        :return instance of Block (sub-)class: the parsed object.
        """
        if names is None:
            names = cls.COLUMN_LABELS
        df = pd.read_csv(io.StringIO(''.join(lines)),
                         *args,
                         names=names,
                         index_col=index_col,
                         header=header,
                         sep=sep,
                         quotechar=quotechar,
                         **kwargs)
        if df.empty:
            return cls(df)
        if cls.ID_COLS is not None:
            df[cls.ID_COLS] -= 1
        if cls.TYPE_COL is not None:
            df[cls.TYPE_COL] -= 1
        if index_col == 0:
            df.index -= 1
        return cls(df)

    def write(self,
              hdl,
              as_block=True,
              columns=None,
              sep=SPACE,
              header=False,
              float_format='%.4f',
              mode='a',
              quotechar=POUND,
              **kwargs):
        """
        Write the data to a file buffer.

        :param hdl `_io.TextIOWrapper` or `_io.StringIO`: write to this handler
        :param as_block `bool`: whether to write the data as a block.
        :param columns list: the labels of the columns to write out.
        :param sep `str`: the separator to use.
        :param header `bool`: whether to write the column names as the header.
        :param float_format `str`: the format to use for floating point numbers.
        :param mode `str`: the mode to use for writing.
        :param quotechar `str`: the quote character to use.
        """
        if columns is None:
            columns = self.COLUMN_LABELS
        if self.empty:
            return
        content = self.NAME + '\n\n' if as_block and self.NAME else ''
        self.index += 1
        if self.TYPE_COL is not None:
            self[self.TYPE_COL] += 1
        if self.ID_COLS is not None:
            self[self.ID_COLS] += 1
        content += self.to_csv(columns=columns,
                               sep=sep,
                               header=header,
                               float_format=float_format,
                               mode=mode,
                               quotechar=quotechar,
                               **kwargs)
        self.index -= 1
        if self.TYPE_COL is not None:
            self[self.TYPE_COL] -= 1
        if self.ID_COLS is not None:
            self[self.ID_COLS] -= 1
        if as_block:
            content += '\n'
        hdl.write(content)


class Box(Block):

    NAME = ''
    LABEL = 'box'
    LO, HI = 'lo', 'hi'
    INDEX = ['x', 'y', 'z']
    ORIGIN = [0, 0, 0]
    LIMIT_CMT = '{limit}_cmt'
    LO_LABEL, HI_LABEL = LIMIT_CMT.format(limit=LO), LIMIT_CMT.format(limit=HI)
    COLUMN_LABELS = [LO, HI, LO_LABEL, HI_LABEL]
    LO_CMT = [x + y for x, y in itertools.product(INDEX, [LO])]
    HI_CMT = [x + y for x, y in itertools.product(INDEX, [HI])]
    FLT_RE = "[+-]?[\d\.\d]+"
    LO_HI = [f'{x}{Block.SPACE}{y}' for x, y in zip(LO_CMT, HI_CMT)]
    RE = re.compile(f"^{FLT_RE}\s+{FLT_RE}\s+({'|'.join(LO_HI)}).*$")

    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['_span']
    _internal_names_set = set(_internal_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._span = None
        self[self.LO_LABEL] = self.LO_CMT
        self[self.HI_LABEL] = self.HI_CMT

    @property
    def span(self):
        """
        Set and cache the span of the box.

        :return: the span of the box.
        :rtype: 'numpy.ndarray'
        """
        if self._span is not None:
            return self._span
        self._span = (self.hi - self.lo).values
        return self._span

    @classmethod
    def fromEdges(cls, edges):
        """
        Box built from these edges and origin.

        :param list: the box edges.
        """
        return cls(data={cls.LO: cls.ORIGIN, cls.HI: edges})

    def write(self, fh, index=False, **kwargs):
        """
        Write the box into the handler.

        :param hdl `_io.TextIOWrapper` or `_io.StringIO`: write to this handler.
        :param index `bool`: whether to write the index.
        """
        super().write(fh, index=index, **kwargs)

    def to_str(self):
        """
        return the box as string.

        :return: the box data in str
        :rtype: st
        """
        return self.to_csv(header=False,
                           lineterminator=' ',
                           index=False,
                           sep=' ')

    def getPoint(self):
        """
        Get a random point within the box.

        :return 'pandas.core.series.Series': the random point within the box.
        """
        point = np.random.rand(3) * self.span
        return point + self.lo

    @property
    def edges(self):
        """
        Get the edges from point list of low and high points.

        :return 12x2x3 numpy.ndarray: 12 edges of the box, and each edge
            contains two points.
        """
        # Three edges starting from the [xlo, ylo, zlo]
        lo_xyzs = np.array([self.lo.values] * 3, dtype=float)
        lo_points = lo_xyzs.copy()
        np.fill_diagonal(lo_points, self.hi.values)
        lo_edges = np.stack((lo_xyzs, lo_points), axis=1)
        # Three edges starting from the [xhi, yhi, zhi]
        hi_xyzs = np.array([self.hi.values] * 3, dtype=float)
        hi_points = hi_xyzs.copy()
        np.fill_diagonal(hi_points, self.lo)
        hi_edges = np.stack((hi_xyzs, hi_points), axis=1)
        # Six edges connecting the open ends of the known edges
        spnts = collections.deque([x[1] for x in lo_edges])
        epnts = collections.deque([x[1] for x in hi_edges])
        epnts.rotate(1)
        oedges = [[x, y] for x, y in zip(spnts, epnts)]
        epnts.rotate(1)
        oedges += [[x, y] for x, y in zip(spnts, epnts)]
        return np.concatenate((lo_edges, hi_edges, np.array(oedges)))


class Mass(Block):
    """
    The masses of the atoms in the system.
    """

    NAME = 'Masses'
    COLUMN_LABELS = ['mass', 'comment']
    LABEL = 'atom types'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._element = None

    def writeCount(self, fh):
        """
        Write the count with the label appended.

        :param hdl `_io.TextIOWrapper` or `_io.StringIO`: write to this handler
        """
        fh.write(f'{self.shape[0]} {self.LABEL}\n')

    @classmethod
    def fromLines(cls, *args, index_col=0, **kwargs):
        """
        Construct a mass instance from a list of lines.

        :param index_col int: Column(s) to use as row label(s)
        :return 'Mass' or subclass instance: the mass.
        """
        return super().fromLines(*args, index_col=index_col, **kwargs)

    @property
    def element(self, rex='.*\s(\w+)\s\w+'):
        """
        Set and cache the element of the atom types.

        :param rex str: the regular expression to extract elements from the
            comment column.
        :return: the element of the atom types.
        :rtype: 'numpy.ndarray'
        """
        if self._element is not None:
            return self._element
        self._element = self.comment.str.extract(rex).values.flatten()
        return self._element


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
    COLUMN_LABELS = [PairCoeff.ENE, 'deg']
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
    XYZU = symbols.XYZU
    TYPE_COL = [TYPE_ID]
    COLUMN_LABELS = [MOL_ID, TYPE_ID, CHARGE] + XYZU
    ID_COLS = [MOL_ID]
    LABEL = 'atoms'

    def mapIds(self, id_map):
        """
        Create a copy of the instance with atom ids mapped to global atom ids.

        :param id_map 'numpy.ndarray': map atom indexes to global atom indexes.
        :return (sub-)class instance: the new instance with global atom ids.
        """
        acopy = self.copy()
        acopy.index = id_map[acopy.index]
        return acopy

    @classmethod
    def concat(cls, objs, ignore_index=True, **kwargs):
        """
        Concatenate the instances and re-index the row from 1.

        :param objs: the instances to concatenate.
        :type objs: list of (sub-)class instances.
        :return: the concatenated data
        :rtype: (sub-)class instances
        """
        if not len(objs):
            return cls()
        return pd.concat(objs, ignore_index=ignore_index, **kwargs)


class Bond(Atom):

    NAME = 'Bonds'
    ID_COLS = [ATOM1, ATOM2]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS
    DEFAULT_DTYPE = int
    LABEL = 'bonds'

    def __init__(self,
                 data=None,
                 type_ids=None,
                 aids=None,
                 dtype=DEFAULT_DTYPE,
                 **kwargs):
        """
        :param data: ndarray, Iterable, dict, or DataFrame
        :type data: the content to create dataframe
        :param type_ids: type ids of the aids
        :type type_ids: list of int
        :param aids: each sublist contains atom ids matching with one type id
        :type aids: list of list
        :param dtype: 'the data type of the Series
        :type dtype: 'type'
        """
        if data is None and type_ids is not None and aids is not None:
            data = [[x] + y for x, y in zip(type_ids, aids)]
        if data is None:
            data = {x: pd.Series(dtype=dtype) for x in self.COLUMN_LABELS}
        super().__init__(data=data, **kwargs)

    def mapIds(self, id_map):
        """
        Create a copy of the instance with atom ids mapped to global atom ids.

        :param id_map 'numpy.ndarray': map atom id columns with global indexes.
        :return (sub-)class instance: the new instance with global atom ids.
        """
        acopy = self.copy()
        acopy[self.ID_COLS] = id_map[acopy[self.ID_COLS]]
        return acopy

    def getPairs(self, step=1):
        """
        Get the atom pairs from each topology connectivity.

        :param step: the step when slicing the atom ids
        :type step: int
        :return: the atom pairs
        :rtype: list of tuple
        """
        slices = slice(None, None, step)
        return [tuple(sorted(x[slices])) for x in self[self.ID_COLS].values]

    def getRigid(self, func):
        """
        Get the rigid topology types.

        :param func: whether this type of topology is rigid
        :type func: callable
        :return: the rigid topology types.
        :rtype: `DataFrame`
        """
        vals = [x for x in self[TYPE_ID].unique() if func(x)]
        return pd.DataFrame({self.NAME: vals})


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
        Get the angles indexes from atom ids.

        :param atom_ids `numpy.ndarray`: each row is atom ids from one angle
        :return Angle: the selected angles matching the input atom ids.
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
        :return int: the index of the angle with the lowest energy.
        """
        return min(self.index, key=lambda x: func(self.loc[x].type_id))


class Dihedral(Bond):

    NAME = 'Dihedrals'
    ID_COLS = [ATOM1, ATOM2, ATOM3, ATOM4]
    COLUMN_LABELS = [TYPE_ID] + ID_COLS
    LABEL = 'dihedrals'

    def getPairs(self, step=3):
        """
        Get the atom pairs from each topology connectivity.

        :param step: the step when slicing the atom ids
        :type step: int
        :return: the atom pairs
        :rtype: list of tuple
        """
        return super(Dihedral, self).getPairs(step=step)


class Improper(Dihedral):

    NAME = 'Impropers'
    LABEL = 'impropers'

    def getPairs(self):
        """
        Get the atom pairs from each topology connectivity.

        :param step: the step when slicing the atom ids
        :type step: int
        :return: the atom pairs
        :rtype: list of tuple
        """
        ids = [itertools.combinations(x, 2) for x in self[self.ID_COLS].values]
        return [tuple(sorted(y)) for x in ids for y in x]

    def getAngles(self):
        """
        Get the atom pairs from each topology connectivity.

        :return: each row contains three angles by one improper angle atoms.
        :rtype: ndarray
        """
        columns = [ATOM2, ATOM1, ATOM4]
        cols = [[x, ATOM3, y] for x, y in itertools.combinations(columns, 2)]
        return np.array([x for x in zip(*[self[x].values for x in cols])])


class Conformer(structure.Conformer):

    @property
    def atoms(self):
        """
        Atoms in the conformer.

        :return `Atom`: information such as atom global ids, molecule ids, atom
            type ids, charges, coordinates.
        """
        atoms = self.GetOwningMol().atoms.mapIds(self.id_map)
        atoms.mol_id = self.gid
        atoms[Atom.XYZU] = self.GetPositions()
        return atoms

    @property
    def bonds(self):
        """
        Bonds in the conformer.

        :return `Bond`: information such as bond ids and bonded atom ids.
        """
        return self.GetOwningMol().bonds.mapIds(self.id_map)

    @property
    def angles(self):
        """
        Angles in the conformer.

        :return `Angle`: information such as angle ids and connected atom ids.
        """
        return self.GetOwningMol().angles.mapIds(self.id_map)

    @property
    def dihedrals(self):
        """
        Dihedral angles in the conformer.

        :return `Dihedral`: information such as dihedral ids and connected atom ids.
        """
        return self.GetOwningMol().dihedrals.mapIds(self.id_map)

    @property
    def impropers(self):
        """
        Improper angles in the conformer.

        :return `Improper`: information such as improper ids and connected atom ids.
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
        self.nbr_charge = collections.defaultdict(float)
        if self.ff is None and self.struct and hasattr(self.struct, 'ff'):
            self.ff = self.struct.ff
        if self.ff is None:
            self.ff = oplsua.get_parser()
        if self.delay:
            return
        self.type()

    def type(self):
        """
        Type atoms and set charges.
        """
        self.typeAtoms()
        self.balanceCharge()

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
    @functools.cache
    def atoms(self):
        """
        The atoms of the molecules.

        :return: Atoms with type ids and charges
        :rtype: Atom
        """
        type_ids = [x.GetIntProp(TYPE_ID) for x in self.GetAtoms()]
        fchrg = [self.ff.charges[x] for x in type_ids]
        aids = [x.GetIdx() for x in self.GetAtoms()]
        nchrg = [self.nbr_charge[x] for x in aids]
        chrg = [sum(x) for x in zip(fchrg, nchrg)]
        return Atom({TYPE_ID: type_ids, Atom.CHARGE: chrg}, index=aids)

    @property
    @functools.cache
    def bonds(self):
        """
        The bonds of the molecule.

        :return Bond: the bond types and bonded atom ids.
        """
        bonds = [x for x in self.GetBonds()]
        type_ids = [self.ff.getMatchedBonds(x)[0].id for x in bonds]
        aids = [[x.GetBeginAtom().GetIdx(),
                 x.GetEndAtom().GetIdx()] for x in bonds]
        return Bond(type_ids=type_ids, aids=aids)

    @property
    @functools.cache
    def angles(self):
        """
        Angle force of the molecules after removal due to improper angles.

        e.g. NH3 if all three H-N-H angles are defined, you cannot control out
        of plane mode.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multi-pole and Gaussian
        Multi-xpole Models

        :return Angle: the angle types and atoms forming each angle.
        """
        angles = [y for x in self.GetAtoms() for y in self.ff.getAngleAtoms(x)]
        type_ids = [self.ff.getMatchedAngles(x)[0].id for x in angles]
        aids = [[y.GetIdx() for y in x] for x in angles]
        angles = Angle(type_ids=type_ids, aids=aids)
        matches = [angles.select(x) for x in self.impropers.getAngles()]
        index = [x.getIndex(lambda x: self.ff.angles[x].ene) for x in matches]
        return angles.drop(index=index)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Dihedral angles of the molecules.

        :return Dihedral: the dihedral types and atoms forming each dihedral.
        """
        dihes = [x for x in self.getDihAtoms()]
        type_ids = [self.ff.getMatchedDihedrals(x)[0].id for x in dihes]
        aids = [[y.GetIdx() for y in x] for x in dihes]
        return Dihedral(type_ids=type_ids, aids=aids)

    @property
    @functools.cache
    def impropers(self):
        """
        Improper angles of the molecules.

        :return Improper: the improper types and atoms forming each improper.
        """
        imprps = [x for x in self.getImproperAtoms()]
        type_ids = [self.ff.getMatchedImpropers(x)[0] for x in imprps]
        aids = [[y.GetIdx() for y in x] for x in imprps]
        return Improper(type_ids=type_ids, aids=aids)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight

    def getRigid(self):
        """
        The bond and angle are rigid during simulation.

        :return DataFrame, DataFrame: the type ids of the rigid bonds and angles
        """
        bnd_types = self.bonds.getRigid(lambda x: self.ff.bonds[x].has_h)
        ang_types = self.angles.getRigid(lambda x: self.ff.angles[x].has_h)
        return bnd_types, ang_types

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


class Struct(structure.Struct, lammpsin.In):

    MolClass = Mol
    DESCR = 'LAMMPS Description # {style}'

    def __init__(self, struct=None, ff=None, options=None, **kwargs):
        """
        :param struct Struct: struct object with moelcules and conformers.
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(struct=struct, **kwargs)
        lammpsin.In.__init__(self, options=options, **kwargs)
        self.ff = ff
        self.total_charge = 0.
        self.atm_types = numpyutils.IntArray()
        self.bnd_types = numpyutils.IntArray()
        self.ang_types = numpyutils.IntArray()
        self.dihe_types = numpyutils.IntArray()
        self.impr_types = numpyutils.IntArray()
        self.initTypeMap()

    def initTypeMap(self):
        """
        Initiate type map.
        """
        self.atm_types = numpyutils.IntArray(max(self.ff.atoms))
        self.bnd_types = numpyutils.IntArray(max(self.ff.bonds))
        self.ang_types = numpyutils.IntArray(max(self.ff.angles))
        self.dihe_types = numpyutils.IntArray(max(self.ff.dihedrals))
        self.impr_types = numpyutils.IntArray(max(self.ff.impropers))

    def addMol(self, mol):
        """
        Add a molecule to the structure.

        :param mol: add this molecule to the structure
        :type mol: Mol
        :return: the added molecule
        :rtype: Mol
        """
        mol = super().addMol(mol)
        self.setTypeMap(mol)
        return mol

    def setTypeMap(self, mol):
        """
        Set the type map for atoms, bonds, angles, dihedrals, and impropers.

        :param mol: add this molecule to the structure
        :type mol: Mol
        """

        atypes = [x.GetIntProp(TYPE_ID) for x in mol.GetAtoms()]
        self.atm_types[atypes] = True
        self.bnd_types[mol.bonds[TYPE_ID]] = True
        self.ang_types[mol.angles[TYPE_ID]] = True
        self.dihe_types[mol.dihedrals[TYPE_ID]] = True
        self.impr_types[mol.impropers[TYPE_ID]] = True

    def writeData(self, nofile=False):
        """
        Write out a LAMMPS datafile or return the content.

        :param nofile bool: return the content as a string if True.
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
            self.box.write(self.hdl)
            # Interaction coefficients
            self.masses.write(self.hdl)
            self.pair_coeffs.write(self.hdl)
            self.bond_coeffs.write(self.hdl)
            self.angle_coeffs.write(self.hdl)
            self.dihedral_coeffs.write(self.hdl)
            self.improper_coeffs.write(self.hdl)
            # Topology details
            self.atoms.write(self.hdl)
            self.bonds.write(self.hdl)
            self.angles.write(self.hdl)
            self.dihedrals.write(self.hdl)
            self.impropers.write(self.hdl)
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
        """
        Atoms in the structure.

        :return `Atom`: information such as atom global ids, molecule ids, atom
            type ids, charges, coordinates.
        """
        data = Atom.concat([x.atoms for x in self.conformer])
        data[TYPE_ID] = self.atm_types.map(data[TYPE_ID])
        return data

    @property
    def bonds(self):
        """
        Bonds in the structure.

        :return `Bond`: information such as bond ids and bonded atom ids.
        """
        bonds = [x.bonds for x in self.conformer if not x.bonds.empty]
        bonds = Bond.concat(bonds, axis=0)
        bonds[TYPE_ID] = self.bnd_types.map(bonds[TYPE_ID])
        return bonds

    @property
    def angles(self):
        """
        Angle in the structure.

        :return `Angle`: information such as angle ids and connected atom ids.
        """
        angles = [x.angles for x in self.conformer if not x.angles.empty]
        angles = Angle.concat(angles, axis=0)
        angles[TYPE_ID] = self.ang_types.map(angles[TYPE_ID])
        return angles

    @property
    def dihedrals(self):
        """
        Dihedral angles in the structure.

        :return `Dihedral`: information such as dihedral ids and connected atom ids.
        """
        dihes = [x.dihedrals for x in self.conformer if not x.dihedrals.empty]
        dihes = Dihedral.concat(dihes, axis=0)
        dihes[TYPE_ID] = self.dihe_types.map(dihes[TYPE_ID])
        return dihes

    @property
    def impropers(self):
        """
        Improper angles in the structure.

        :return `Improper`: information such as improper ids and connected atom ids.
        """
        imprps = [x.impropers for x in self.conformer if not x.impropers.empty]
        imprps = Improper.concat(imprps, axis=0)
        imprps[TYPE_ID] = self.impr_types.map(imprps[TYPE_ID])
        return imprps

    @property
    def masses(self):
        """
        Atom masses.

        :return `Mass`: mass of each type of atom.
        """
        masses = [self.ff.atoms[x] for x in self.atm_types.on]
        masses = Mass([[x.mass, f" {x.description} {x.symbol} {x.id} "]
                       for x in masses])
        return masses

    @property
    def pair_coeffs(self):
        """
        Non-bonded atom pair coefficients.

        :return `PairCoeff`: the interaction between non-bond atoms.
        """
        vdws = [self.ff.vdws[x] for x in self.atm_types.on]
        return PairCoeff([[x.ene, x.dist] for x in vdws])

    @property
    def bond_coeffs(self):
        """
        Bond coefficients.

        :return `BondCoeff`: the interaction between bonded atoms.
        """
        bonds = [self.ff.bonds[x] for x in self.bnd_types.on]
        return BondCoeff([[x.ene, x.dist] for x in bonds])

    @property
    def angle_coeffs(self):
        """
        Angle coefficients.

        :return `AngleCoeff`: the three-atom angle interaction coefficients
        """
        angles = [self.ff.angles[x] for x in self.ang_types.on]
        return AngleCoeff([[x.ene, x.deg] for x in angles])

    @property
    def dihedral_coeffs(self):
        """
        Dihedral coefficients.

        :return `DihedralCoeff`: the four-atom torsion interaction coefficients
        """
        dihes = [self.ff.dihedrals[x] for x in self.dihe_types.on]
        return DihedralCoeff([x.params for x in dihes])

    @property
    def improper_coeffs(self):
        """
        Improper coefficients.

        :return `ImproperCoeff`: the four-atom improper interaction coefficients
        """
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

    def writeRun(self, *arg, **kwarg):
        """
        Write command to further equilibrate the system with molecules
        information considered.
        """
        if self.options.rigid_bond is None and self.options.rigid_angle is None:
            self.options.rigid_bond, self.options.rigid_angle = self.getRigid()
        single_molecule = self.conformer_total == 1
        small_molecule = self.atom_total < 100
        single_point_energy = not self.options.temp
        testing = any([single_molecule, small_molecule, single_point_energy])
        super().writeRun(*arg, testing=testing, **kwarg)

    def getRigid(self):
        """
        Get the rigid bond and angle types.

        :return: the rigid bond and angle types
        :rtype: str, str
        """
        data = [x.getRigid() for x in self.molecules]
        bonds, angles = list(map(list, zip(*data)))
        bonds = Bond.concat([x for x in bonds if not x.empty])
        angles = Angle.concat([x for x in angles if not x.empty])
        bond_types = self.bnd_types.map(bonds.values.flatten()) + 1
        angle_types = self.ang_types.map(angles.values.flatten()) + 1
        return [' '.join(map(str, x)) for x in [bond_types, angle_types]]

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        return not np.isclose(self.atoms.charge, 0, 0.001).any()

    def getWarnings(self):
        """
        Get warnings for the structure.

        :return generator of str: the warnings on structure checking.
        """
        net_charge = round(self.atoms.charge.sum(), 4)
        if net_charge:
            yield f'The system has a net charge of {net_charge:.4f}'
        min_span = self.box.span.min()
        if min_span < self.options.lj_cut * 2:
            yield f'The minimum box span ({min_span:.2f} {symbols.ANGSTROM})' \
                  f' is smaller than {self.options.lj_cut * 2:.2f} ' \
                  f'{symbols.ANGSTROM} (Lennard-Jones Cutoff x 2) '


class DataFileReader(lammpsin.In):
    """
    LAMMPS Data file reader.
    """
    BLOCK_CLASSES = [
        Mass, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff, ImproperCoeff,
        Atom, Bond, Angle, Dihedral, Improper
    ]
    BLOCK_NAMES = [x.NAME for x in BLOCK_CLASSES]
    BLOCK_LABELS = [x.LABEL for x in BLOCK_CLASSES]
    NAME_RE = re.compile(f"^{'|'.join(BLOCK_NAMES)}$")
    COUNT_RE = re.compile(f"^[0-9]+\s+({'|'.join(BLOCK_LABELS)})$")
    DESCR = Struct.DESCR.split(symbols.POUND)[0]

    def __init__(self, data_file=None, contents=None, delay=False):
        """
        :param data_file str: data file with path
        :param contents `bytes`: parse the contents if data_file not provided.
        """
        self.data_file = data_file
        self.contents = contents
        self.lines = None
        self.name = {}
        if delay:
            return
        self.read()
        self.index()

    def read(self):
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

    def index(self):
        """
        Index the lines by block markers, and Parse the description section for
        topo counts and type counts.
        """
        names = {}
        for idx, line in enumerate(self.lines):
            match = self.NAME_RE.match(line)
            if not match:
                continue
            # The block name occupies one lien and there is one empty line below
            names[match.group()] = idx + 2

        counts = {}
        for line in self.lines[:min(names.values())]:
            match = self.COUNT_RE.match(line)
            if not match:
                continue
            # 'atoms': 1620, 'bonds': 1593, 'angles': 1566 ...
            # 'atom types': 7, 'bond types': 6, 'angle types': 5 ...
            counts[match.group(1)] = int(line.split(match.group(1))[0])

        for block_class in self.BLOCK_CLASSES:
            if block_class.NAME not in names:
                continue
            idx = names[block_class.NAME]
            count = counts[block_class.LABEL]
            self.name[block_class.NAME] = slice(idx, idx + count)

        lines = self.lines[:min(names.values())]
        # 'xlo xhi': [-7.12, 35.44], 'ylo yhi': [-7.53, 34.26], ..
        box_lines = [i for i, x in enumerate(lines) if Box.RE.match(x)]
        self.name[Box.LABEL] = slice(min(box_lines), max(box_lines) + 1)

    @property
    @functools.cache
    def box(self):
        """
        Parse the box section.

        :return `Box`: the box
        """
        return self.fromLines(Box)

    @property
    @functools.cache
    def masses(self):
        """
        Parse the mass section for masses and elements.

        :return `Mass`: the masses of atoms.
        """
        return self.fromLines(Mass)

    @property
    @functools.cache
    def pair_coeffs(self):
        """
        Paser the pair coefficient section.

        :return `PairCoeff`: the pair coefficients between non-bonded atoms.
        """
        return self.fromLines(PairCoeff)

    @property
    @functools.cache
    def atoms(self):
        """
        Parse the atom section.

        :return `Atom`: the atom information such as atom id, molecule id,
            type id, charge, position, etc.
        """
        return self.fromLines(Atom)

    @property
    @functools.cache
    def bonds(self):
        """
        Parse the atom section for atom id and molecule id.

        :return `Bond`: the bond information such as id, type id, and bonded
            atom ids.
        """
        return self.fromLines(Bond)

    @property
    @functools.cache
    def angles(self):
        """
        Parse the angle section for angle id and constructing atoms.

        :return `Angle`: the angle information such as id, type id, and atom ids
            in the angle.
        """
        return self.fromLines(Angle)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.

        :return `Dihedral`: the dihedral angle information such as id, type id,
            and atom ids in the dihedral angle.
        """
        return self.fromLines(Dihedral)

    @property
    @functools.cache
    def impropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.

        :return `Improper`: the improper angle information such as id, type id,
            and atom ids in the improper angle.
        """
        return self.fromLines(Improper)

    def fromLines(self, BlockClass):
        """
        Parse a block of lines from the datafile.

        :param BlockClass: the class to handle a block.
        :return BlockClass: the parsed block.
        """
        name = BlockClass.NAME if BlockClass.NAME else BlockClass.LABEL
        if name not in self.name:
            return BlockClass.fromLines([])
        lines = self.lines[self.name[name]]
        return BlockClass.fromLines(lines)

    @property
    def elements(self, name='element'):
        """
        The elements of all atoms.

        :param name: the name of the element column.
        :type name: str
        :return: the element dataframe with atom ids
        :rtype: `pd.DataFrame`
        """
        data = self.masses.element[self.atoms.type_id]
        return pd.DataFrame(data, index=self.atoms.index, columns=[name])

    @property
    @functools.cache
    def molecules(self):
        """
        The atom ids grouped by molecules.

        :return: keys are molecule ids and values are atom global ids.
        :rtype: dict
        """
        mols = collections.defaultdict(list)
        for gid, mid, in self.atoms.mol_id.items():
            mols[mid].append(gid)
        return dict(mols)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.masses.mass[self.atoms.type_id].sum()

    mw = molecular_weight


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
        kwargs = dict(index=range(dists.index.max() + 1), fill_value=-1)
        radii = dists.reindex(**kwargs).values.tolist()
        radii = np.full((len(radii), len(radii)), radii, dtype='float16')
        radii *= radii.transpose()
        radii = np.sqrt(radii)
        radii *= pow(2, 1 / 6) * cls.SCALE
        radii[radii < cls.MIN_DIST] = cls.MIN_DIST
        obj = np.asarray(radii).view(cls)
        kwargs = dict(index=range(atypes.index.max() + 1), fill_value=-1)
        obj.id_map = atypes.reindex(**kwargs).values
        return obj
