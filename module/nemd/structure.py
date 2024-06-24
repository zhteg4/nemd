"""
This module handles conformer, molecule and structure.
"""
import rdkit
import functools
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors

from nemd import symbols
from nemd import rdkitutils


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
        Set up the conformer global id, global atoms ids, and owning molecule.

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

        :return list of int: the atom ids of this conformer.
        """
        return np.where(self.id_map != -1)[0].tolist()

    @property
    @functools.cache
    def gids(self):
        """
        Return the global atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return self.id_map[self.id_map != -1].tolist()

    def HasOwningMol(self):
        """
        Returns whether this conformer belongs to a molecule.

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


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of Chem.rdchem.Mol with additional attributes and methods.
    """

    ConfClass = Conformer

    def __init__(self, mol=None, struct=None, delay=False, **kwargs):
        """
        :param struct 'Mol': the molecule instance
        :param struct 'Struct': owning structure
        :param delay bool: customization is delayed for later setup or testing.
        """
        # conformers in super(Mol, self).GetConformers() are rebuilt
        super().__init__(mol, **kwargs)
        self.struct = struct
        self.delay = delay
        self.confs = []
        if self.delay:
            return
        if mol is None:
            return
        self.setUp(mol.GetConformers())

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
            self.confs.append(conf)
            gid += self.GetNumAtoms()

    def GetConformer(self, id=0):
        """
        Get the conformer of the molecule.

        :param id int: the conformer id to get.
        :return `Conformer`: the selected conformer.
        """
        return self.confs[id]

    def AddConformer(self, conf, **kwargs):
        """
        Add conformer to the molecule.

        :param conf `rdkit.Chem.rdchem.Conformer`: the conformer to add.
        :return `Conformer`: the added conformer.
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
        return self.confs[:]

    def EmbedMolecule(self, **kwargs):
        """
        Embed the molecule to generate a conformer.
        """
        # EmbedMolecule clear previous conformers, and only add one.
        with rdkitutils.rdkit_warnings_ignored():
            rdkit.Chem.AllChem.EmbedMolecule(self, **kwargs)
        self.confs.clear()
        self.setUp(super().GetConformers())

    @classmethod
    def MolFromSmiles(cls, smiles, united=True, **kwargs):
        """
        Create a molecule from SMILES.

        :param smiles str: the SMILES string.
        :param united bool: hide keep Hydrogen atoms in CH, CH3, CH3, and CH4.
        :return `Mol`: the molecule instance.
        """

        mol = rdkit.Chem.MolFromSmiles(smiles)
        if not united:
            return cls(mol, **kwargs)

        # Hide Hs in CH, CH3, CH3, and CH4
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != symbols.CARBON or atom.GetIsAromatic():
                continue
            atom.SetIntProp(symbols.IMPLICIT_H, atom.GetNumImplicitHs())
            atom.SetNoImplicit(True)

        # FIXME: support different chiralities for monomers
        chiral = rdkit.Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for chirality in chiral:
            # CIP stereochemistry assignment for the molecule’s atoms (R/S)
            # and double bonds (Z/E)
            mol.GetAtomWithIdx(chirality[0]).SetProp('_CIPCode', 'R')

        return cls(rdkit.Chem.AddHs(mol), **kwargs)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return rdkit.Chem.Descriptors.ExactMolWt(self)

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

    def __init__(self, struct=None):
        """
        :param struct 'Struct': the structure with molecules.
        """
        self.molecules = []
        self.density = None
        if struct is None:
            return
        for mol in struct.molecules:
            self.addMol(mol)
        self.finalize()

    @classmethod
    def fromMols(cls, mols, *args, **kwargs):
        """
        Create structure instance from molecules.

        :param mols list of 'Chem.rdchem.Mol': the molecules to be added.
        :return 'Struct': the structure containing the molecules.
        """
        struct = cls(*args, **kwargs)
        for mol in mols:
            struct.addMol(mol)
        struct.finalize()
        return struct

    def addMol(self, mol):
        """
        Initialize molecules and conformers with id and map set.

        :param mol 'Mol': the molecule to be added.
        :return 'Mol': the added molecule.
        """
        mol = self.MolClass(mol, struct=self)
        self.molecules.append(mol)
        return mol

    def finalize(self):
        """
        Finalize the structure after all molecules are added.
        """
        pass

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

        :return list of `Conformer`: the conformers of all molecules.
        """
        return [x for y in self.molecules for x in y.GetConformers()]

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

        :return 'pandas.core.frame.DataFrame': the positions of all conformers.
        """
        return pd.concat([
            pd.DataFrame(x.GetPositions(), index=x.gids, columns=symbols.XYZU)
            for x in self.conformers
        ])

    @property
    def conformer_total(self):
        """
        Get the total number of all conformers.

        :return int: the total number of all conformers.
        """
        return sum([x.GetNumConformers() for x in self.molecules])
