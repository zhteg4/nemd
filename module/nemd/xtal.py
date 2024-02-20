# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Builder crystals.
"""
import crystals
from rdkit import Chem
from nemd import rdkitutils
from nemd import constants as nconstants


class CrystalBuilder(object):

    def __init__(self,
                 name,
                 dim=nconstants.ONE_ONE_ONE,
                 scale_factor=nconstants.ONE_ONE_ONE):
        self.name = name
        self.dim = dim
        self.scale_factor = scale_factor
        self.scell = None

    def run(self):
        self.setSuperCell()

    def setSuperCell(self):
        cell = crystals.Crystal.from_database(self.name)
        vect = [x * y for x, y in zip(cell.lattice_vectors, self.scale_factor)]
        ucell_xtal = crystals.Crystal(cell, vect)
        self.scell = ucell_xtal.supercell(*self.dim)

    def getMol(self):
        atoms = [x for x in self.scell.atoms]
        atoms = sorted(atoms, key=lambda x: tuple(x.coords_fractional))
        mol = Chem.Mol()
        emol = Chem.EditableMol(mol)
        idxs = [emol.AddAtom(Chem.rdchem.Atom(x.atomic_number)) for x in atoms]
        mol = emol.GetMol()
        cfm = Chem.rdchem.Conformer(mol.GetNumAtoms())
        [cfm.SetAtomPosition(x, atoms[x].coords_cartesian) for x in idxs]
        mol.AddConformer(cfm)
        return rdkitutils.Mol(mol,
                              lattice_parameters=self.scell.lattice_parameters,
                              dimensions=self.scell.dimensions)
