"""
This module handles molecular topology and structural editing.
"""
import rdkit
import functools
import numpy as np

from nemd import logutils
from nemd import rdkitutils

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


class Conformer(rdkit.Chem.rdchem.Conformer):
    """
    A subclass of Chem.rdchem.Conformer with additional attributes and methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mol = None
        self.id_map = None
        self.gid = 1

    def setUp(self, mol, cid=1, gid=1):
        """
        Set up the conformer global ids.

        :param mol `Chem.rdchem.Mol`: the original molecule.
        :param cid int: the conformer gid to start with.
        :param gid int: the starting global id.
        """
        self.mol = mol
        self.gid = cid
        if self.mol is None:
            return
        gids = [x for x in range(gid, gid + self.mol.GetNumAtoms())]
        id_map = {x.GetIdx(): y for x, y in zip(self.mol.GetAtoms(), gids)}
        max_gid = max(id_map.keys()) + 1
        self.id_map = np.array([id_map.get(x, -1) for x in range(max_gid)])

    @property
    @functools.cache
    def aids(self):
        """
        Return the atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return list(np.where(self.id_map != -1)[0])

    def HasOwningMol(self):
        """
        Returns whether or not this conformer belongs to a molecule.

        :return `bool`: the molecule this conformer belongs to.
        """
        return bool(self.GetOwningMol())

    def GetOwningMol(self):
        """
        Get the Mol that owns this conformer.

        :return `rdkit.Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        return self.mol

    def setPositions(self, xyz):
        """
        Reset the positions of the atoms to the original xyz coordinates.

        :return xyz np.ndarray: the xyz coordinates of the molecule.
        """
        for id in range(xyz.shape[0]):
            self.SetAtomPosition(id, xyz[id, :])

    def centroid(self, aids=None, ignoreHs=False):
        """
        Compute the centroid of the whole conformer ar the selected atoms.

        :param atom_ids list: the selected atom ids
        :param ignoreHs bool: whether to ignore Hs in the calculation.
        :return np.ndarray: the centroid of the selected atoms.
        """
        weights = None
        if aids is not None:
            bv = rdkit.DataStructs.ExplicitBitVect(self.GetNumAtoms())
            bv.SetBitsFromList(aids)
            weights = rdkit.rdBase._vectd()
            weights.extend(bv.ToList())
        return rdkit.Chem.rdMolTransforms.ComputeCentroid(self,
                                                          weights=weights,
                                                          ignoreHs=ignoreHs)

    def translate(self, vect):
        """
        Do translation on this conformer using this vector.

        :param vect 'numpy.ndarray': translational vector
        """
        mtrx = np.identity(4)
        mtrx[:-1, 3] = vect
        rdkit.Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def setBondLength(self, bonded, val):
        """
        Set bond length of the given dihedral.

        :param bonded tuple of int: the bonded atom indices.
        :param val val: the bond distance.
        """
        rdkit.Chem.rdMolTransforms.SetBondLength(self, *bonded, val)


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of Chem.rdchem.Mol with additional attributes and methods.
    """

    ConfClass = Conformer

    def __init__(self, *args, struct=None, ff=None, delay=False, **kwargs):
        """
        :param struct 'Struct': owning structure
        :param ff 'OplsParser': the force field class.
        :delay bool: customization is delayed for later setup or testing.
        """
        super().__init__(*args, **kwargs)
        self.struct = struct
        self.ff = ff
        self.delay = delay
        self.conf_id = 0
        self.confs = {}
        if delay:
            return
        if args:
            self.setUp(args[0].GetConformers())

    def setUp(self, confs, cid=1, gid=1):
        """
        Set up the conformers including global ids and references.

        :param confs `Chem.rdchem.Conformers`: the conformers from the original
            molecule.
        :param cid int: the conformer gid to start with.
        :param gid int: the starting global id.
        """
        if self.struct:
            cid, gid = self.struct.getIds()
        for cid, conf in enumerate(confs, start=cid):
            conf = self.ConfClass(conf)
            conf.setUp(self, cid=cid, gid=gid)
            self.confs[conf.GetId()] = conf
            gid += self.GetNumAtoms()

    def setConformerId(self, conf_id):
        """
        Set the selected conformer id.

        :param conf_id int: the conformer id to select.
        """
        self.conf_id = conf_id

    def GetConformer(self, conf_id=None):
        """
        Get the conformer of the molecule.

        :param conf_id int: the conformer id to get.
        :return `rdkitChem.rdchem.Conformer`: the selected conformer.
        """
        if conf_id is None:
            conf_id = self.conf_id
        return self.confs[conf_id]

    def AddConformer(self, conf, **kwargs):
        """
        Add conformer to the molecule.

        :param conf `rdkit.Chem.rdchem.Conformer`: the conformer to add.
        :return `rdkit.Chem.rdchem.Conformer`: the added conformer.
        """
        # AddConformer handles the super().GetOwningMol()
        id = super(Mol, self).AddConformer(conf, **kwargs)
        self.setUp([super(Mol, self).GetConformer(id)])
        return self.GetConformer(id)

    def GetConformers(self):
        """
        Get the conformers of the molecule.

        :return list of conformers: the conformers of the molecule.
        """
        return list(self.confs.values())

    def EmbedMolecule(self, **kwargs):
        """
        Embed the molecule to generate a conformer.
        """
        # EmbedMolecule clear previous conformers, and only add one.
        with rdkitutils.rdkit_warnings_ignored():
            rdkit.Chem.AllChem.EmbedMolecule(self, **kwargs)
        self.confs.clear()
        self.setUp(super().GetConformers())

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight

    @property
    def atom_total(self):
        """
        The total number of atoms in all conformers.

        :return int: the total number of atoms in all conformers.
        """
        return self.GetNumAtoms() * self.GetNumConformers()


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """

    MolClass = Mol

    def __init__(self, struct=None, ff=None):
        """
        :param struct 'Struct': the structure with molecules.
        :param ff 'OplsParser': the force field class.
        """
        self.ff = ff
        self.mols = {}
        self.density = None
        if struct is None:
            return
        for mol in struct.mols.values():
            self.addMol(mol)

    @classmethod
    def fromMols(cls, mols, *args, **kwargs):
        """
        Create structure instance from molecules.

        :param mols list of 'Chem.rdchem.Mol': the molecules to be added.
        """
        struct = cls(*args, **kwargs)
        for mol in mols:
            struct.addMol(mol)
        return struct

    def addMol(self, mol, mol_id=0):
        """
        Initialize molecules and conformers with id and map set.

        :param mol 'Mol': the molecule to be added.
        :param mol_id int: the starting id of the molecules.
        """
        if self.mols:
            mol_id = max(self.mols.keys()) + 1
        mol = self.MolClass(mol, struct=self, ff=self.ff)
        self.mols[mol_id] = mol
        return mol_id

    def getIds(self, cid=1, gid=1):
        """
        Get the global ids to start with.

        :param cid int: the conformer gid to start with.
        :param gid int: the starting global id.
        :retrun int, int: the conformer gid, the global atom id.
        """
        if self.conformers:
            cid = max([x.gid for x in self.conformers]) + 1
            gid = max([x.id_map.max() for x in self.conformers]) + 1
        return cid, gid

    @property
    def conformers(self):
        """
        Return all conformers of all molecules.

        :return list of Chem.rdchem.Conformer: the conformers of all
            molecules.
        """
        return [x for y in self.molecules for x in y.GetConformers()]

    @property
    def molecules(self):
        """
        Return all conformers of all molecules.

        :return list of Chem.rdchem.Conformer: the conformers of all
            molecules.
        """
        return [x for x in self.mols.values()]

    @property
    def atoms(self):
        """
        Return atoms of molecules.

        Note: the len() of these atoms is different atom_total as atom_toal
        includes atoms from all conformers.

        :return list of Chem.rdchem.Atom: the atoms from all molecules.
        """
        return [y for x in self.molecules for y in x.GetAtoms()]

    @property
    def atom_total(self):
        """
        The total number of atoms in all conformers across all molecules.

        :return int: the total number of atoms in all conformers.
        """
        return sum([x.atom_total for x in self.molecules])

    def getPositions(self):
        """
        Get the positions of all conformers.

        :return np.ndarray: the positions of all conformers.
        """
        return np.concatenate([x.GetPositions() for x in self.conformers])

    @property
    def conformer_total(self):
        """
        Get the total number of all conformers.

        :return int: the total number of all conformers.
        """
        return sum([x.GetNumConformers() for x in self.molecules])
