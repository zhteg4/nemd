"""
This module handles molecular topology and structural editing.
"""
import rdkit
import functools
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs

from nemd import logutils

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
    A subclass of rdkit.Chem.rdchem.Conformer with additional attributes and methods.
    """

    def __init__(self, *args, mol=None, **kwargs):
        """
        :param mol `rdkit.Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        super().__init__(*args, **kwargs)
        self.mol = mol

    def setGids(self, atom_ids, start_gid):
        """
        Set the global ids of the atoms in this conformer.

        :param atom_ids list of int: the atom ids in this conformer.
        :param start_gid int: the starting global id.
        """
        gids = [x for x in range(start_gid, start_gid + len(atom_ids))]
        id_map = {x: y for x, y in zip(atom_ids, gids)}
        max_gid = max(atom_ids) + 1
        self.id_map = np.array([id_map.get(x, -1) for x in range(max_gid)])

    @property
    @functools.cache
    def aids(self):
        """
        Return the atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return list(np.where(self.id_map != -1)[0])

    def SetOwningMol(self, mol):
        """
        Set the Mol that owns this conformer.
        """
        self.mol = mol

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
            bv = DataStructs.ExplicitBitVect(self.GetNumAtoms())
            bv.SetBitsFromList(aids)
            weights = rdBase._vectd()
            weights.extend(bv.ToList())

        return Chem.rdMolTransforms.ComputeCentroid(self,
                                                    weights=weights,
                                                    ignoreHs=ignoreHs)

    def translate(self, vect):
        """
        Do translation on this conformer using this vector.

        :param vect 'numpy.ndarray': translational vector
        """
        mtrx = np.identity(4)
        mtrx[:-1, 3] = vect
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def setBondLength(self, bonded, val):
        """
        Set bond length of the given dihedral.

        :param bonded tuple of int: the bonded atom indices.
        :param val val: the bond distance.
        """
        Chem.rdMolTransforms.SetBondLength(self, *bonded, val)


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, ff=None, delay=False, **kwargs):
        """
        :param ff 'OplsParser': the force field class.
        :delay bool: customization is delayed for later setup or testing.
        """
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.delay = delay
        self.conf_id = 0
        self.confs = None

    def setIdMap(self, start_gid):
        """
        Set the atom map num and the global ids of the atoms in all conformers
        of the molecule.

        :param gid int: the starting global id.
        :return start_gid int: the next starting global id.
        """
        atom_ids = [x.GetIdx() for x in self.GetAtoms()]
        for id, atom in enumerate(self.GetAtoms(), start=1):
            atom.SetAtomMapNum(id)
        for conf in self.GetConformers():
            conf.setGids(atom_ids, start_gid)
            start_gid += self.GetNumAtoms()
        return start_gid

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
        :return `rdkit.Chem.rdchem.Conformer`: the selected conformer.
        """
        if conf_id is None:
            conf_id = self.conf_id
        if self.confs is None:
            self.initConformers()
        return self.confs[conf_id]

    def GetConformers(self):
        """
        Get the conformers of the molecule.

        :return list of conformers: the conformers of the molecule.
        """
        if self.confs is None:
            self.initConformers()
        return list(self.confs.values())

    def initConformers(self, ConfClass=Conformer):
        """
        Set the conformers of the molecule.

        :param ConfClass (sub)class 'rdkit.Chem.rdchem.Conformer': the conformer
            class to instantiate conformers.
        """
        confs = super().GetConformers()
        self.confs = {x.GetId(): ConfClass(x, mol=self) for x in confs}

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

    def __init__(self, mols, MolClass=Mol, ff=None, **kwargs):
        """
        :param mols list of rdkit.Chem.rdchem.Mol: the molecules to be handled.
        :param MolClass subclass of 'rdkit.Chem.rdchem.Mol': the customized
            molecule class
        :param ff 'OplsParser': the force field class.
        """
        # Initialize molecules and conformers
        self.mols = {
            i: MolClass(x, ff=ff)
            for i, x in enumerate(mols, start=1)
        }
        # Set conf_id for each conformer
        for conf_id, conf in enumerate(self.conformers, start=1):
            conf.SetId(conf_id)
        # Set gids for atoms in each conformer
        start_gid = 1
        for mol in self.molecules:
            start_gid = mol.setIdMap(start_gid=start_gid)
        self.density = None

    @property
    def conformers(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all
            molecules.
        """
        return [x for y in self.molecules for x in y.GetConformers()]

    @property
    def molecules(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all
            molecules.
        """
        return [x for x in self.mols.values()]

    @property
    def atoms(self):
        """
        Return atoms of molecules.

        Note: the len() of these atoms is different atom_total as atom_toal
        includes atoms from all conformers.

        :return list of rdkit.Chem.rdchem.Atom: the atoms from all molecules.
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

    def getNumConformers(self):
        """
        Get the total number of all conformers.

        :return np.ndarray: the total number of all conformers.
        """
        return sum([x.GetNumConformers() for x in self.molecules])
