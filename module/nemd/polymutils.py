import copy
import functools
import collections
import numpy as np
import networkx as nx
from rdkit import Chem

from nemd import pnames
from nemd import oplsua
from nemd import symbols
from nemd import logutils
from nemd import structure
from nemd import rdkitutils
from nemd import structutils
from nemd import parserutils

FLAG_CRU = 'cru'
FLAG_CRU_NUM = '-cru_num'
FLAG_MOL_NUM = '-mol_num'

FLAG_BUFFER = '-buffer'
FLAG_SUBSTRUCT = '-substruct'
FLAG_NO_MINIMIZE = '-no_minimize'
FLAG_RIGID_BOND = '-rigid_bond'
FLAG_RIGID_ANGLE = '-rigid_angle'


class Mol(structure.Mol, logutils.Base):
    """
    Class to hold a regular molecule or a polymer built from monomers.
    """

    IMPLICIT_H = oplsua.IMPLICIT_H
    MONO_ATOM_IDX = 'mono_atom_idx'
    CAP = 'cap'
    HT = 'ht'
    POLYM_HT = pnames.POLYM_HT
    IS_MONO = pnames.IS_MONO
    MONO_ID = pnames.MONO_ID

    def __init__(self,
                 cru,
                 cru_num,
                 mol_num,
                 options=None,
                 delay=False,
                 **kwargs):
        """
        :param cru str: the smiles string for monomer
        :param cru_num int: the number of monomers per polymer
        :param mol_num int: the number of molecules of this type of polymer
        :param ff 'OplsParser': the force field class.
        :param options 'argparse.Namespace': command line options
        :delay bool: if True, the object is initialized without building the
            polymer (base class __init__ is called in the setUp method).
        """
        logutils.Base.__init__(self, **kwargs)
        self.cru = cru
        self.cru_num = cru_num
        self.mol_num = mol_num
        self.options = options
        self.polym_Hs = None
        self.box = None
        self.cru_mol = None
        self.smiles = None
        if delay:
            return
        self.build()

    def build(self):
        """
        Main method to build one polymer.
        """
        self.setCruMol()
        self.markMonomer()
        self.polymerize()
        self.embedMol()
        self.setConformers()

    def setCruMol(self):
        """
        Set monomer mol based on the input smiles.
        """
        self.cru_mol = structure.Mol.MolFromSmiles(self.cru)

    def markMonomer(self):
        """
        Marker monomer original atom indexes, head/tail and capping atoms.
        """

        capping = [
            x for x in self.cru_mol.GetAtoms()
            if x.GetSymbol() == symbols.WILD_CARD
        ]
        is_mono = True if capping else False
        self.cru_mol.SetBoolProp(self.IS_MONO, is_mono)
        [x.SetBoolProp(self.CAP, True) for x in capping]
        [
            y.SetBoolProp(self.HT, True) for x in capping
            for y in x.GetNeighbors()
        ]
        [
            x.SetIntProp(self.MONO_ATOM_IDX, x.GetIdx())
            for x in self.cru_mol.GetAtoms()
        ]

    def polymerize(self):
        """
        Polymerize from the monomer mol.
        """
        if not self.cru_mol.GetBoolProp(self.IS_MONO):
            super().__init__(self.cru_mol)
            return
        # Duplicate and index monomers
        mols = []
        for mono_id in range(1, self.cru_num + 1):
            mol = copy.copy(self.cru_mol)
            mols.append(mol)
            for atom in mol.GetAtoms():
                atom.SetIntProp(self.MONO_ID, mono_id)
        # Combine monomers into one molecule
        combo = mols[0]
        for mol in mols[1:]:
            combo = Chem.CombineMols(combo, mol)
        # Search head/tail atoms
        capping_atoms = [
            x for x in combo.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
        ]
        ht_atoms = [x.GetNeighbors()[0] for x in capping_atoms]
        ht_atom_idxs = [x.GetIdx() for x in ht_atoms]
        for polym_ht in [ht_atom_idxs[0], ht_atom_idxs[-1]]:
            atom = combo.GetAtomWithIdx(polym_ht)
            atom.SetBoolProp(self.POLYM_HT, True)
            atom.SetIntProp(self.IMPLICIT_H,
                            atom.GetIntProp(self.IMPLICIT_H) + 1)
        # Create bonds between monomers
        edcombo = Chem.EditableMol(combo)
        for t_atom_idx, h_atom_idx in zip(
                ht_atom_idxs[1:-1:2],
                ht_atom_idxs[2::2],
        ):
            edcombo.AddBond(t_atom_idx,
                            h_atom_idx,
                            order=Chem.rdchem.BondType.SINGLE)
        polym = edcombo.GetMol()
        # Delete capping atoms
        orgin_atom_num = None
        while orgin_atom_num != polym.GetNumAtoms():
            orgin_atom_num = polym.GetNumAtoms()
            polym = Chem.DeleteSubstructs(
                polym, Chem.MolFromSmiles(symbols.WILD_CARD))
        super().__init__(polym)
        self.log(f"Polymer SMILES: {Chem.MolToSmiles(self)}")

    def embedMol(self):
        """
        Embed the molecule with coordinates.
        """

        if self.GetNumAtoms() <= 200:
            with rdkitutils.CaptureLogger() as logs:
                # Mg+2 triggers
                # WARNING UFFTYPER: Warning: hybridization set to SP3 for atom 0
                # ERROR UFFTYPER: Unrecognized charge state for atom: 0
                self.EmbedMolecule(useRandomCoords=True,
                                   randomSeed=self.options.seed)
                [self.log_debug(f'{x} {y}') for x, y in logs.items()]
            Chem.GetSymmSSSR(self)
            return

        trans_conf = Conformer(self, self.cru_mol, options=self.options)
        trans_conf.run()

    def setConformers(self):
        """
        Set multiple conformers based on the first one.
        """
        tpl_conf = self.GetConformer(0)
        for conf_id in range(1, self.mol_num):
            conf = structure.Conformer(tpl_conf)
            self.AddConformer(conf, assignId=True)


class Conformer(object):
    """
    Conformer coordinate assignment.
    """

    CAP = Mol.CAP
    MONO_ATOM_IDX = Mol.MONO_ATOM_IDX
    MONO_ID = Mol.MONO_ID

    def __init__(self, polym, original_cru_mol, options=None):
        """
        :param polym 'rdkit.Chem.rdchem.Mol': the polymer to set conformer
        :param original_cru_mol 'rdkit.Chem.rdchem.Mol': the monomer mol
            constructing the polymer
        :param options 'argparse.Namespace': command line options.
        """
        self.polym = polym
        self.original_cru_mol = original_cru_mol
        self.options = options
        self.relax_dir = '_relax'
        self.data_file = 'polym.data'
        self.conformer = None
        self.cru_mol = None

    def run(self):
        """
        Main method set the conformer.
        """

        self.setCruMol()
        self.setCruConformer()
        self.setCruBackbone()
        self.transAndRotate()
        self.rotateSideGroups()
        self.setXYZAndVect()
        self.setConformer()

    def setCruMol(self):
        """
        Set monomer mol with elements tuned.
        """
        cru_mol = copy.copy(self.original_cru_mol)
        atoms = [
            x for x in cru_mol.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
        ]
        neighbors = [x.GetNeighbors()[0] for x in atoms[::-1]]
        for atom, catom in zip(atoms, neighbors):
            atom.SetAtomicNum(catom.GetAtomicNum())
            atom.SetBoolProp(self.CAP, True)
        self.cru_mol = structutils.GrownMol(cru_mol, delay=True)
        self.cru_mol.setGraph()

    def setCruConformer(self):
        """
        Set the cru conformer.
        """
        self.cru_mol.EmbedMolecule()
        self.cru_conf = self.cru_mol.GetConformer(0)

    def setCruBackbone(self):
        """
        Set the cru backbone atom ids.

        :raise ValueError: if the number of capping atoms is not 2.
        """
        cap_idxs = [
            x.GetIdx() for x in self.cru_mol.GetAtoms() if x.HasProp(self.CAP)
        ]
        if len(cap_idxs) != 2:
            raise ValueError(f'{len(cap_idxs)} capping atoms are found.')
        self.cru_bk_aids = nx.shortest_path(self.cru_mol.graph, *cap_idxs)

    def transAndRotate(self):
        """
        Set trans-conformer with translation and rotation.
        """

        for dihe in zip(self.cru_bk_aids[:-3], self.cru_bk_aids[1:-2],
                        self.cru_bk_aids[2:-1], self.cru_bk_aids[3:]):
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conf, *dihe, 180)

        cntrd = self.cru_conf.centroid(aids=self.cru_bk_aids)
        self.cru_conf.translate(-np.array(cntrd))
        abc_norms = self.getABCVectors()
        abc_targeted = np.eye(3)
        self.cru_conf.rotate(abc_norms, abc_targeted)

    def getABCVectors(self):
        """
        Get the a, b, c vectors of the molecule.

        :return 3x3 'numpy.ndarray': a vector is parallel with the backbone,
            b and c vectors are perpendicular with the backbone.
        """

        def get_norm(vect):
            """
            Get the normalized vector.

            :param vect 'numpy.ndarray': input vector
            :return 'numpy.ndarray': normalized vector
            """
            vect /= np.linalg.norm(vect)
            return vect

        bh_xyzs = np.array(
            [self.cru_conf.GetAtomPosition(x) for x in self.cru_bk_aids])
        bvectors = (bh_xyzs[1:, :] - bh_xyzs[:-1, :])
        nc_vec = get_norm(bvectors[::2].mean(axis=0))
        nm_mvec = get_norm(bvectors[1::2].mean(axis=0))
        avect = get_norm(nc_vec + nm_mvec).reshape(1, -1)
        bvect = get_norm(nc_vec - nm_mvec).reshape(1, -1)
        cvect = get_norm(np.cross(avect, bvect)).reshape(1, -1)
        abc_norms = np.concatenate([avect, bvect, cvect], axis=0)
        return abc_norms

    def rotateSideGroups(self):
        """
        Rotate the bond between the backbone and side group.
        """

        def get_other_atom(aid1, aid2):
            """
            Get one atom id that is bonded to aid1 beyond aid2.

            :param aid1 int: The neighbors of this atom is searched.
            :param aid2 int: This is excluded from the neighbor search
            :return one atom id that is bonded to aid1 beyond aid2:
            """
            aids = self.cru_mol.GetAtomWithIdx(aid1).GetNeighbors()
            aids = [x.GetIdx() for x in aids]
            aids.remove(aid2)
            return aids[0]

        # Search for atom pairs connecting the backbone and the side group
        bonds = self.cru_mol.GetBonds()
        bonded_aids = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in bonds]
        bk_ids = set(self.cru_bk_aids)
        side_aids = [
            x for x in bonded_aids if len(bk_ids.intersection(x)) == 1
        ]
        side_aids = [x if x[0] in bk_ids else reversed(x) for x in side_aids]
        # Get the dihedral atom ids that move the side group
        side_dihes = []
        for id2, id3 in side_aids:
            id1 = get_other_atom(id2, id3)
            id4 = get_other_atom(id3, id2)
            side_dihes.append([id1, id2, id3, id4])
        for dihe in side_dihes:
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conf, *dihe, 90)

    def setXYZAndVect(self):
        """
        Set the XYZ of the non-capping atoms and translational vector, preparing
        for the monomer coordinate shifting.
        """

        cap_ht = [x for x in self.cru_mol.GetAtoms() if x.HasProp(self.CAP)]
        cap_ht = [(x.GetIdx(), x.GetNeighbors()[0].GetIdx()) for x in cap_ht]
        middle_points = np.array([
            np.mean([self.cru_conf.GetAtomPosition(y) for y in x], axis=0)
            for x in cap_ht
        ])

        self.vector = middle_points[1, :] - middle_points[0, :]
        self.xyzs = {
            x.GetIntProp(self.MONO_ATOM_IDX):
            np.array(self.cru_conf.GetAtomPosition(x.GetIdx()))
            for x in self.cru_mol.GetAtoms() if x.HasProp(self.MONO_ATOM_IDX)
        }

    def setConformer(self):
        """
        Build and set the conformer.
        """
        mono_id_atoms = collections.defaultdict(list)
        for atom in self.polym.GetAtoms():
            mono_id_atoms[atom.GetIntProp(self.MONO_ID)].append(atom)

        conformer = structure.Conformer(self.polym.GetNumAtoms())
        for mono_id, atoms in mono_id_atoms.items():
            vect = mono_id * self.vector
            for atom in atoms:
                mono_atom_id = atom.GetIntProp(self.MONO_ATOM_IDX)
                xyz = self.xyzs[mono_atom_id] + vect
                conformer.SetAtomPosition(atom.GetIdx(), xyz)
        self.polym.AddConformer(conformer, assignId=True)
        Chem.GetSymmSSSR(self.polym)

    @classmethod
    def write(cls, mol, filename):
        """
        Write the polymer and monomer into sdf files.

        :param mol 'rdkit.Chem.rdchem.Mol': The molecule to write out
        :param filename str: The file path to write into
        """

        with Chem.SDWriter(filename) as fh:
            try:
                mono_atom_idxs = [
                    x.GetIntProp(cls.MONO_ATOM_IDX) for x in mol.GetAtoms()
                ]
            except KeyError:
                fh.write(mol)
                return
            mol.SetProps([cls.MONO_ATOM_IDX])
            mol.SetProp(cls.MONO_ATOM_IDX, ' '.join(map(str, mono_atom_idxs)))
            fh.write(mol)

    @classmethod
    def read(cls, filename):
        """
        Read molecule from file path.

        :param filename str: the file path to read molecule from.
        :return 'rdkit.Chem.rdchem.Mol': The molecule with properties.
        """
        suppl = Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
        mol = next(suppl)
        Chem.GetSymmSSSR(mol)
        try:
            mono_atom_idxs = mol.GetProp(cls.MONO_ATOM_IDX).split()
        except KeyError:
            return mol
        for atom, mono_atom_idx in zip(mol.GetAtoms(), mono_atom_idxs):
            atom.SetProp(cls.MONO_ATOM_IDX, mono_atom_idx)
        return mol


class Validator:

    def __init__(self, options):
        """
        param options: Command line options.
        """
        self.options = options

    def run(self):
        """
        Main method to run the validation.
        """
        self.cruNum()
        self.molNum()

    def cruNum(self):
        """
        Validate (or set) the number of repeat units.
        """
        if self.options.cru_num is None:
            self.options.cru_num = [1] * len(self.options.cru)
            return
        if len(self.options.cru_num) == len(self.options.cru):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.cru_num)} cru found.')

    def molNum(self):
        """
        Validate (or set) the number of molecules.
        """
        if self.options.mol_num is None:
            self.options.mol_num = [1] * len(self.options.cru_num)
            return
        if len(self.options.mol_num) == len(self.options.cru_num):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.mol_num)} molecules found.')


def add_arguments(parser):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser.add_argument(
        FLAG_CRU,
        metavar=FLAG_CRU.upper(),
        type=functools.partial(parserutils.type_monomer_smiles,
                               allow_mol=True),
        nargs='+',
        help='SMILES of the constitutional repeat unit (monomer)')
    parser.add_argument(
        FLAG_CRU_NUM,
        metavar=FLAG_CRU_NUM[1:].upper(),
        type=parserutils.type_positive_int,
        nargs='+',
        help='Number of constitutional repeat unit per polymer')
    parser.add_argument(FLAG_MOL_NUM,
                        metavar=FLAG_MOL_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        nargs='+',
                        help='Number of molecules in the amorphous cell')
    parser.add_argument(FLAG_NO_MINIMIZE,
                        action='store_true',
                        help='Skip the structure minimization step.')
    parser.add_argument(
        FLAG_RIGID_BOND,
        metavar=FLAG_RIGID_BOND[1:].upper(),
        type=parserutils.type_positive_int,
        nargs='+',
        help='The bond types whose lengths are fixed during the simulation.')
    parser.add_argument(
        FLAG_RIGID_ANGLE,
        metavar=FLAG_RIGID_ANGLE[1:].upper(),
        type=parserutils.type_positive_int,
        nargs='+',
        help='The angles of these types remain fixed during the simulation.')
    parser.add_argument(
        FLAG_BUFFER,
        metavar=FLAG_BUFFER[1:].upper(),
        type=parserutils.type_positive_float,
        help='The buffer distance between molecules in the grid cell.')
    parser.add_argument(
        FLAG_SUBSTRUCT,
        metavar='SMILES:VALUE',
        type=parserutils.type_substruct,
        help='Select a substructure and set the coordinates accordingly.')
    parserutils.add_md_arguments(parser)
    return parser
