# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles molecular topology and structural editing.
"""
import math
import rdkit
import scipy
import warnings
import itertools
import functools
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs
from scipy.spatial.transform import Rotation

from nemd import traj
from nemd import oplsua
from nemd import pnames
from nemd import logutils

EDGES = 'edges'
WEIGHT = 'weight'

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


def getGraph(mol):
    """
    Get the networkx graph on the input molecule.
    :param mol `rdkit.Chem.rdchem.Mol`: the input molecule with/without bonds

    :return `networkx.classes.graph.Graph`: graph with nodes and edges.
    """
    graph = nx.Graph()
    edges = [(
        x.GetBeginAtom().GetIdx(),
        x.GetEndAtom().GetIdx(),
    ) for x in mol.GetBonds()]
    if not edges:
        # When bonds don't exist, just add the atom.
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx())
        return graph
    # When bonds exist, add edges and the associated atoms, assuming atoms in
    # one molecule are bonded.
    graph.add_edges_from(edges)
    for edge in edges:
        for idx in range(2):
            node = graph.nodes[edge[idx]]
            try:
                node[EDGES].append(edge)
            except KeyError:
                node[EDGES] = [edge]
    return graph


def findPath(graph=None, mol=None, source=None, target=None, **kwarg):
    """
    Find the path in a molecule.

    :param graph 'networkx.classes.graph.Graph': molecular networkx graph
    :param mol `rdkit.Chem.rdchem.Mol`: molecule to find path on
    :param source int: the input source node
    :param target int: the input target node
    :return int, int, list: source node, target node, and the path inbetween
    """

    if graph is None:
        graph = getGraph(mol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shortest_path = nx.shortest_path(graph,
                                         source=source,
                                         target=target,
                                         **kwarg)

    if target is not None:
        shortest_path = {target: shortest_path}
    if source is not None:
        shortest_path = {source: shortest_path}
    path_length, path = -1, None
    for a_source_node, target_path in shortest_path.items():
        for a_target_node, a_path in target_path.items():
            if path_length >= len(a_path):
                continue
            source_node = a_source_node
            target_node = a_target_node
            path = a_path
            path_length = len(a_path)
    return source_node, target_node, path


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

    def GetOwningMol(self):
        """
        Get the Mol that owns this conformer.

        :return `rdkit.Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        return self.mol

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
        return bool(self.mol)

    def centroid(self, aids=None):
        """
        Compute the centroid of the whole conformer ar the selected atoms.

        :param atom_ids list: the selected atom ids
        """
        if aids is None:
            return Chem.rdMolTransforms.ComputeCentroid(self)

        bv = DataStructs.ExplicitBitVect(self.GetNumAtoms())
        bv.SetBitsFromList(aids)
        weights = rdBase._vectd()
        weights.extend(bv.ToList())
        return Chem.rdMolTransforms.ComputeCentroid(self,
                                                    weights=weights,
                                                    ignoreHs=False)

    def translate(self, vect):
        """
        Do translation on this conformer using this vector.

        :param vect 'numpy.ndarray': translational vector
        """
        mtrx = np.identity(4)
        mtrx[:-1, 3] = vect
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def rotateRandomly(self):
        """
        Randomly rotate the conformer.

        NOTE: the random state is set according to the numpy random seed.
        """
        mtrx = np.identity(4)
        seed = np.random.randint(0, 2**32 - 1)
        mtrx[:-1, :-1] = Rotation.random(random_state=seed).as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def rotate(self, ivect, tvect):
        """
        Rotate the conformer by three initial vectors and three target vectors.

        :param ivect 3x3 'numpy.ndarray': Each row is one initial vector
        :param tvect 3x3 'numpy.ndarray': Each row is one corresponding target vector
        """
        rotation, rmsd = Rotation.align_vectors(tvect, ivect)
        mtrx = np.identity(4)
        mtrx[:-1, :-1] = rotation.as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)


class ConfError(RuntimeError):
    """
    When max number of the failure for this conformer has been reached.
    """
    pass


class PackedConf(Conformer):

    MAX_TRIAL_PER_CONF = 10000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_map = None
        self.frm = None
        self.dcell = None
        self.extg_gids = None
        self.radii = None
        self.excluded = None

    def setReferences(self):
        """
        Set the references to the conformer.
        """
        self.id_map = self.mol.id_map
        self.radii = self.mol.radii
        self.excluded = self.mol.excluded
        self.frm = self.mol.frm
        self.dcell = self.mol.dcell
        self.extg_gids = self.mol.extg_gids

    @property
    def gids(self):
        """
        Return the global atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return self.id_map[self.GetId()]

    def setConformer(self, max_trial=MAX_TRIAL_PER_CONF):
        """
        Place molecules one molecule into the cell without clash.

        :param max_trial int: the max trial number for each molecule to be placed
            into the cell.
        :raise MolError: if the conformer always has clashes with the existing
            atoms in the cell after the maximum trial.
        """
        for _ in range(max_trial):
            self.translate(-np.array(self.centroid()))
            self.rotateRandomly()
            self.translate(self.frm.getPoint())
            self.frm.loc[self.gids] = self.GetPositions()
            if self.hasClashes():
                continue
            self.extg_gids.update(self.gids)
            # Only update the distance cell after one molecule successful
            # placed into the cell as only inter-molecular clashes are
            # checked for packed cell.
            self.dcell.setUp()
            return
        raise ConfError

    def hasClashes(self):
        """
        Whether the conformer has any clashes with the existing atoms in the cell.

        :param bool: the conformer has clashes or not.
        """
        for id, row in self.frm.loc[self.gids].iterrows():
            clashes = self.dcell.getClashes(row,
                                            included=self.extg_gids,
                                            radii=self.radii,
                                            excluded=self.excluded)
            if clashes:
                return True
        return False


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf_id = 0
        self.confs = None

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
        if self.confs is not None:
            return [x for x in self.confs.values()]
        self.initConformers()
        return [x for x in self.confs.values()]

    def initConformers(self, ConfClass=Conformer):
        """
        Set the conformers of the molecule.

        :param ConfClass (sub)class 'rdkit.Chem.rdchem.Conformer': the conformer
            class to instantiate conformers.
        """
        confs = super().GetConformers()
        self.confs = {i: ConfClass(x, mol=self) for i, x in enumerate(confs)}


class GriddedMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol to handle gridded conformers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # size = xyz span + buffer
        self.buffer = np.array([4, 4, 4])
        # The number of molecules per box edge
        self.mol_num = np.array([1, 1, 1])
        # The xyz shift within one box
        self.vecs = []

    @property
    def size(self):
        """
        Return the box size of this molecule.
        """
        # Grid layout assumes all conformers from one molecule are the same
        xyzs = self.GetConformer().GetPositions()
        return (xyzs.max(axis=0) - xyzs.min(axis=0)) + self.buffer

    @property
    def box_num(self):
        """
        Return the number of boxes needed to place all molecules.
        """
        return math.ceil(self.GetNumConformers() / np.prod(self.mol_num))

    def setMolNumPerEdge(self, box):
        """
        Set the number of molecules per edge of the box.

        :param box np.ndarray: the box size to place this molecule in.
        """
        self.mol_num = np.floor(box / self.size).astype(int)

    def setVecs(self, box):
        """
        Set the translational vectors for this molecule so that this molecule
        can be placed in the given box.

        :param box np.ndarray: the box size to place this molecule in.
        """
        ptc = [np.linspace(-0.5, 0.5, x, endpoint=False) for x in self.mol_num]
        ptc = [x - x.mean() for x in ptc]
        self.vecs = [
            x * box for x in itertools.product(*[[y for y in x] for x in ptc])
        ]

    def setConformers(self, vector):
        """
        Set the conformers of this molecule based on the shifting vector.

        :param vector np.ndarray: the translational vector to move the conformer
            by multiple boxes distances.
        """
        mol_num_per_box = np.prod(self.mol_num)
        for idx in range(min([self.GetNumConformers(), mol_num_per_box])):
            vecs = vector + self.vecs[idx]
            self.GetConformer().translate(vecs)
            self.conf_id += 1


class PackedMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, ff=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.id_map = None
        self.radii = None
        self.excluded = None
        self.frm = None
        self.dcell = None
        self.extg_gids = None

    def initConformers(self, ConfClass=PackedConf):
        """
        See parant class for details.
        """
        return super().initConformers(ConfClass=ConfClass)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """

    def __init__(self, mols, MolClass=Mol, **kwargs):
        self.mols = {i: MolClass(x) for i, x in enumerate(mols, start=1)}
        for mol_id, conf in enumerate(self.conformers, start=1):
            conf.SetId(mol_id)

    @property
    def conformers(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all molecules.
        """
        return [x for y in self.molecules for x in y.GetConformers()]

    @property
    def molecules(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all molecules.
        """
        return [x for x in self.mols.values()]


class GriddedStruct(Struct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    def __init__(self, *args, MolClass=GriddedMol, **kwargs):
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.box = np.zeros([3])

    def run(self):
        """
        Set conformers for all molecules.
        """
        self.setBox()
        self.setVectors()
        self.setConformers()

    def setBox(self):
        """
        Set the box as the maximum size over all molecules.
        """
        self.box = np.array([x.size for x in self.molecules]).max(axis=0)

    def setVectors(self):
        """
        Set translational vectors based on the box for all molecules.
        """
        for mol in self.molecules:
            mol.setMolNumPerEdge(self.box)
            mol.setVecs(self.box)

    def setConformers(self):
        """
        Set coordinates.
        """
        # vectors shifts molecules by the largest box size
        box_total = sum(x.box_num for x in self.molecules)
        idxs = range(math.ceil(math.pow(box_total, 1. / 3)))
        vectors = [x * self.box for x in itertools.product(idxs, idxs, idxs)]
        # boxes are filled in random order with all molecules in random order
        molecules = self.molecules[:]
        while molecules:
            mol = np.random.choice(molecules)
            np.random.shuffle(vectors)
            vector = vectors.pop()
            mol.setConformers(vector)
            if mol.conf_id == mol.GetNumConformers():
                molecules.remove(mol)


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class PackedCell(Struct):
    """
    Pack molecules by random rotation and translation.
    """

    MAX_TRIAL_PER_DENSITY = 100

    def __init__(self,
                 *args,
                 MolClass=PackedMol,
                 ff=None,
                 options=None,
                 **kwargs):
        """
        :param polymers 'Polymer': one polymer object for each type
        :param options 'argparse.Namespace': command line options
        """
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        MolClass = functools.partial(MolClass, ff=ff)
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.ff = ff
        self.options = options
        self.extg_gids = set()

    def run(self):
        """
        Create amorphous cell by randomly placing molecules with random
        orientations.
        """
        self.setBoxes()
        self.setDataReader()
        self.setFrameAndDcell()
        self.setMolStructRefs()
        self.placeMols()

    def runWithDensity(self, density):
        """
        Create amorphous cell of the target density by randomly placing
        molecules with random orientations.

        NOTE: the final density of the output cell may be smaller than the
        target if the max number of trial attempt is reached.

        :param density float: the target density
        """
        self.density = density
        self.setBoxes()
        self.setFrameAndDcell()
        self.setReferences()
        self.placeMols()

    def setBoxes(self):
        """
        Set periodic boundary box size.
        """
        weight = sum(x.mw * x.GetNumConformers() for x in self.mols.values())
        vol = weight / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = [0, edge, 0, edge, 0, edge]
        log_debug(f'Cubic box of size {edge:.2f} angstrom is created.')

    def setDataReader(self):
        """
        Set data reader with clash parameters.
        """
        lmw = oplsua.LammpsData(self.mols, ff=self.ff, options=self.options)
        contents = lmw.writeData(nofile=True)
        self.df_reader = oplsua.DataFileReader(contents=contents)
        self.df_reader.run()
        self.df_reader.setClashParams()

    def setFrameAndDcell(self):
        """
        Set the trajectory frame and distance cell.
        """
        index = [atom.id for atom in self.df_reader.atoms.values()]
        xyz = [atom.xyz for atom in self.df_reader.atoms.values()]
        self.frm = traj.Frame(xyz=xyz, index=index, box=self.box)
        self.dcell = traj.DistanceCell(self.frm)
        self.dcell.setUp()

    def setReferences(self):
        """
        Set references to all molecular and conformers.
        """
        for mol in self.mols.values():
            mol.id_map = self.df_reader.mols
            mol.radii = self.df_reader.radii
            mol.excluded = self.df_reader.excluded
            mol.frm = self.frm
            mol.dcell = self.dcell
            mol.extg_gids = self.extg_gids
            for conf in mol.GetConformers():
                conf.setReferences()

    def placeMols(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """
        trial_id, mol_num = 1, len(self.conformers)
        tenth, threshold, = mol_num / 10., 0
        for trial_id in range(1, max_trial + 1):
            self.extg_gids.clear()
            for conf_id, conf in enumerate(self.conformers):
                try:
                    conf.setConformer()
                except ConfError:
                    log_debug(f'{trial_id} trail fails. '
                              f'(Only {conf_id} / {mol_num} '
                              f'molecules placed in the cell.)')
                    break

                if conf_id < threshold:
                    continue
                conf_num = conf_id + 1
                new_line = "" if conf_id == mol_num else ", [!n]"
                log_debug(f"{int(conf_num / mol_num * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)

            # All molecules successfully placed (no break)
            return
        raise DensityError


class GrownConf(PackedConf):
    pass


class GrownMol(PackedMol):

    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    PATT = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
    IS_MONO = pnames.IS_MONO
    MONO_ID = pnames.MONO_ID
    POLYM_HT = pnames.POLYM_HT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = None
        self.graph = getGraph(self)
        self.rotatable_bonds = self.GetSubstructMatches(self.PATT,
                                                        maxMatches=1000000)

    def initConformers(self, ConfClass=GrownConf):
        """
        See parant class for details.
        """
        return super().initConformers(ConfClass=ConfClass)

    def run(self):
        """
        Main method for fragmentation and conformer search.
        """
        self.fragmentize()
        self.readData()
        self.setDCellParams()
        self.setCoords()
        self.setFrm()
        self.setConformer()

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        self.addNxtFrags()
        self.setPreFrags()
        self.setInitAtomIds()

    def addNxtFrags(self):
        """
        Starting from the initial fragment, keep fragmentizing the current
        fragment and adding the newly generated ones to be fragmentized until
        no fragments can be further fragmentized.
        """
        self.ifrag = Fragment([], self)
        to_be_fragmentized = self.ifrag.setFrags()
        while (to_be_fragmentized):
            frag = to_be_fragmentized.pop(0)
            nfrags = frag.setFrags()
            to_be_fragmentized += nfrags

    def setPreFrags(self):
        """
        Set previous fragment.
        """
        all_frags = self.fragments()
        for frag in all_frags:
            for nfrag in frag.nfrags:
                nfrag.pfrag = frag

    def fragments(self):
        """
        Return all fragments.

        :return list: each of the item is one fragment.
        """
        all_frags = []
        if self.ifrag is None:
            return all_frags
        nfrags = [self.ifrag]
        while (nfrags):
            all_frags += nfrags
            nfrags = [y for x in nfrags for y in x.nfrags]
        return all_frags

    def getNumFrags(self):
        """
        Return the number of the total fragments.

        :return int: number of the total fragments.
        """
        return len(self.fragments())

    def setInitAtomIds(self):
        """
        Set initial atom ids that don't belong to any fragments.
        """
        frags = self.fragments()
        aids = [y for x in frags for y in x.aids]
        aids_set = set(aids)
        assert len(aids) == len(aids_set)
        self.extg_aids = set(
            [x for x in range(self.GetNumAtoms()) if x not in aids_set])

    def findPolymPair(self):
        """
        If the molecule is built from momomers, the atom pairs from
        selected from the first and last monomers.

        :return list or iterator of int tuple: each tuple is an atom id pair
        """
        if not self.HasProp(self.IS_MONO) or not self.GetBoolProp(
                self.IS_MONO):
            return [(None, None)]

        ht_mono_ids = {
            x.GetProp(self.MONO_ID): []
            for x in self.GetAtoms() if x.HasProp(self.POLYM_HT)
        }
        for atom in self.GetAtoms():
            try:
                ht_mono_ids[atom.GetProp(self.MONO_ID)].append(atom.GetIdx())
            except KeyError:
                pass

        st_atoms = list(ht_mono_ids.values())
        sources = st_atoms[0]
        targets = [y for x in st_atoms[1:] for y in x]

        return itertools.product(sources, targets)

    def findPath(self, source=None, target=None):
        """
        Find the shortest path between source and target. If source and target
        are not provided, shortest paths between all pairs are computed and the
        long path is returned.

        :param source int: the atom id that serves as the source.
        :param target int: the atom id that serves as the target.
        :return list of ints: the atom ids that form the shortest path.
        """

        return findPath(self.graph, source=source, target=target)

    def isRotatable(self, bond):
        """
        Whether the bond between the two atoms is rotatable.

        :param bond list or tuple of two ints: the atom ids of two bonded atoms
        :return bool: Whether the bond is rotatable.
        """

        in_ring = self.GetBondBetweenAtoms(*bond).IsInRing()
        single = tuple(sorted(bond)) in self.rotatable_bonds
        return not in_ring and single

    def getSwingAtoms(self, *dihe):
        """
        Get the swing atoms when the dihedral angle changes.

        :param dihe list of four ints: the atom ids that form a dihedral angle
        :return list of ints: the swing atom ids when the dihedral angle changes.
        """
        conf = self.GetConformer()
        oxyz = conf.GetPositions()
        oval = Chem.rdMolTransforms.GetDihedralDeg(conf, *dihe)
        Chem.rdMolTransforms.SetDihedralDeg(conf, *dihe, oval + 5)
        xyz = conf.GetPositions()
        changed = np.isclose(oxyz, xyz)
        Chem.rdMolTransforms.SetDihedralDeg(conf, *dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def copy(self, conf):
        """
        Copy the current FragMol object and set the new conformer.
        NOTE: dihedral value candidates, existing atom ids, and fragment references
        are copied. Other attributes such as the graph, rotatable bonds, and frames
        are just referred to the original object.

        :param conf 'rdkit.Chem.rdchem.Conformer': the new conformer.
        """
        fmol = GrownMol(self)
        fmol.conf = conf
        fmol.ifrag = self.ifrag.copyInit(fmol=fmol) if self.ifrag else None
        fmol.extg_aids = self.extg_aids.copy()
        fmol.frm = self.frm
        fmol.graph = self.graph
        fmol.rotatable_bonds = self.rotatable_bonds
        return fmol

    def setGlobalAtomIds(self, gids):
        """
        Set Global atom ids for each fragment and existing atoms.
        """
        self.gids = gids
        id_map = {x.GetIdx(): y for x, y in zip(self.GetAtoms(), self.gids)}
        self.extg_gids = set([id_map[x] for x in self.extg_aids])
        for frag in self.fragments():
            frag.gids = [id_map[x] for x in frag.aids]

    @property
    def molecule_id(self):
        """
        Return molecule id

        :return int: the molecule id.
        """
        return self.conf.GetId()


class GrownStruct(PackedCell):

    MAX_TRIAL_PER_DENSITY = 10

    def __init__(self, *args, MolClass=GrownMol, **kwargs):
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.fmols = {}
        self.failed_num = 0  # The failed attempts in growing molecules
        self.mol_num = None  # the last reported growing molecule number

    def placeMols(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """
        self.fragmentize()
        self.setDCellParams()
        self.setCoords()
        self.setFrm()
        self.setConformer()

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        for mol in self.mols.values():
            mol.fragmentize()

        self.fmols = {}
        for id, mol in self.mols.items():
            mol.conf = None  # conformer doesn't support copy
            for conf in mol.GetConformers():
                fmol = mol.copy(conf)
                mol_id = conf.GetId()
                fmol.setGlobalAtomIds(self.df_reader.mols[mol_id])
                self.fmols[mol_id] = fmol
        total_frag_num = sum([x.getNumFrags() for x in self.fmols.values()])
        log_debug(f"{total_frag_num} fragments in total.")

    def setDCellParams(self):
        """
        Set distance cell parameters.
        """

        # memory saving flaot16 to regular float32
        self.max_clash_dist = float(self.df_reader.radii.max())
        # Using [0][1][2] as the cell, atoms in [0] and [2], are at least
        # Separated by 1 max_clash_dist, meaning no clashes.
        self.cell_cut = self.max_clash_dist

    def setCoords(self):
        """
        Set conformer coordinates from data file.
        """

        for conf in self.conformers:
            mol = conf.GetOwningMol()
            aids = self.df_reader.mols[conf.GetId()]
            for aid, atom in zip(aids, mol.GetAtoms()):
                xyz = self.df_reader.atoms[aid].xyz
                conf.SetAtomPosition(atom.GetIdx(), np.array(xyz))

    def setFrm(self):
        """
        Set traj frame.
        """
        xyz = np.array([x.xyz for x in self.df_reader.atoms.values()])
        self.frm = traj.Frame(xyz=xyz, box=self.box)

    def setConformer(self):
        """
        Set conformer coordinates without clashes.
        """
        frags = [x.ifrag for x in self.fmols.values()]
        self.setInitFrm(frags)
        self.setDcell()
        log_debug(f'Placing {len(frags)} initiators into the cell...')

        tenth, threshold, = len(frags) / 10., 0
        for index, frag in enumerate(frags, start=1):
            self.placeInitFrag(frag)
            if index >= threshold:
                new_line = "" if index == len(frags) else ", [!n]"
                log_debug(f"{int(index / len(frags) * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)
        self.logInitFragsPlaced(frags)

        while frags:
            frag = frags.pop(0)
            if not frag.dihe:
                # ifrag without dihe means rigid body
                continue
            while frag.vals:
                frag.setDihedralDeg()
                self.updateFrm()
                if self.hasClashes(frag.gids):
                    continue
                # Successfully grew one fragment
                frags += frag.nfrags
                self.add(frag.gids)
                self.reportStatus(frags)
                break
            else:
                frags, success = self.backMove(frag, frags)
                if not success:
                    # The molecule has grown to a dead end (no break)
                    self.failed_num += 1
                    frags[0].resetVals()
                    # The method backmove() deletes some extg_gids
                    self.dcell.setGraph(len(self.mols))
                    self.placeInitFrag(frags[0])
                    self.reportRelocation(frags[0])
            log_debug(f'{len(self.dcell.extg_gids)} atoms placed.')

    def setInitFrm(self, frags):
        """
        Set the traj frame for initiators.

        :param frags list of 'fragments.Fragment': fragment from each molecule
        """
        data = np.full((len(frags), 3), np.inf)
        index = [x.fmol.molecule_id for x in frags]
        self.init_tf = traj.Frame(xyz=data, index=index, box=self.box)

    def setDcell(self):
        """
        Set distance cell for neighbor atom and graph for voids searching.
        """
        self.updateFrm()
        self.dcell = traj.DistanceCell(frm=self.frm, cut=self.cell_cut)
        self.dcell.setUp()
        self.dcell.setGraph(len(self.mols))

    def updateFrm(self):
        """
        Update the coordinate frame based on the current conformer.
        """
        pos = [x.GetPositions() for x in self.conformers]
        self.frm.loc[:] = np.concatenate(pos, axis=0)

    def placeInitFrag(self, frag):
        """
        Place the initiator fragment into the cell with random position, random
        orientation, and large separation.

        :param frag 'fragments.Fragment': the fragment to place

        :raise ValueError: when no void to place the initiator fragment of the
            dead molecule.
        """

        self.dcell.rmClashNodes()
        points = self.dcell.getVoids()
        for point in points:
            conf = frag.fmol.conf
            centroid = np.array(conf.centroid(aids=list(frag.fmol.extg_aids)))
            conf.translate(-centroid)
            conf.rotateRandomly()
            conf.translate(point)
            self.frm.loc[frag.fmol.gids] = conf.GetPositions()

            if self.hasClashes(frag.fmol.extg_gids):
                continue
            # Only update the distance cell after one molecule successful
            # placed into the cell as only inter-molecular clashes are
            # checked for packed cell.
            self.add(list(frag.fmol.extg_gids))
            self.init_tf.loc[frag.fmol.molecule_id] = point
            return

        with open('placeInitFrag.xyz', 'w') as out_fh:
            self.frm.write(out_fh,
                           dreader=self.df_reader,
                           visible=list(self.dcell.extg_gids),
                           points=points)
        raise ValueError(f'Failed to relocate the dead molecule. '
                         f'({len(self.dcell.extg_gids)}/{len(self.mols)})')

    def hasClashes(self, gids):
        """
        Whether the atoms has clashes with existing atoms in the cell.

        :param gids list: golabal atom ids to check clashes against.
        :return bool: True if clashes are found.
        """
        frag_rows = [self.frm.loc[x] for x in gids]
        for row in frag_rows:
            clashes = self.dcell.getClashes(row,
                                            included=self.dcell.extg_gids,
                                            radii=self.df_reader.radii,
                                            excluded=self.df_reader.excluded)
            if clashes:
                return True
        return False

    def add(self, gids):
        """
        Update trajectory frame, add atoms to the atom cell and existing record.

        :param gids list: gids of the atoms to be added
        """
        self.updateFrm()
        self.dcell.atomCellUpdate(gids)
        self.dcell.addGids(gids)

    def remove(self, gids):
        """
        Remove atoms from the atom cell and existing record.

        :param gids list: gids of the atoms to be removed
        """
        self.dcell.atomCellRemove(gids)
        self.dcell.removeGids(gids)

    def reportStatus(self, frags):
        """
        Report the growing and failed molecule status.

        :param frags list of 'fragments.Fragment': the growing fragments
        """

        cur_mol_num = len(set([x.fmol.molecule_id for x in frags]))
        if cur_mol_num == self.mol_num:
            # No change of the growing molecule number from previous report
            return

        self.mol_num = cur_mol_num
        finished_num = len(self.mols) - self.mol_num
        log_debug(f'{finished_num} finished; {self.failed_num} failed.')
        return cur_mol_num

    def logInitFragsPlaced(self, frags):
        """
        Log the initiator fragments status after the first placements.

        :param frags list of 'fragments.Fragment': the initiator of each frag
            has been placed into the cell.
        """

        log_debug(f'{len(frags)} initiators have been placed into the cell.')
        if len(self.mols) == 1:
            return
        log_debug(
            f'({self.init_tf.pairDists().min():.2f} as the minimum pair distance)'
        )

    def backMove(self, frag, frags):
        """
        Back move fragment so that the obstacle can be walked around later.

        :param frag 'fragments.Fragment': fragment to perform back move
        :param frags list: growing fragments
        """
        # 1）Find the previous fragment with available dihedral candidates.
        pfrag = frag.getPreAvailFrag()
        found = bool(pfrag)
        frag = pfrag if found else frag.fmol.ifrag
        # 2）Find the next fragments who have been placed into the cell.
        nxt_frags = frag.getNxtFrags()
        [x.resetVals() for x in nxt_frags]
        ratom_gids = [y for x in nxt_frags for y in x.gids]
        if not found:
            ratom_gids += frag.fmol.extg_gids
        self.remove(ratom_gids)
        # 3）Fragment after the next fragments were added to the growing
        # frags before this backmove step.
        nnxt_frags = [y for x in nxt_frags for y in x.nfrags]
        frags = [frag] + [x for x in frags if x not in nnxt_frags]
        log_debug(f"{len(self.dcell.extg_gids)}, {len(frag.vals)}: {frag}")
        return frags, found


class Fragment:

    def __repr__(self):
        return f"{self.dihe}: {self.aids}"

    def __init__(self, dihe, fmol):
        """
        :param dihe list of dihedral atom ids: the dihedral that changes the
            atom position in this fragment.
        :param fmol 'fragments.FragMol': the FragMol that this fragment belongs to
        """
        self.dihe = dihe
        self.fmol = fmol
        self.aids = []
        self.pfrag = None
        self.nfrags = []
        self.vals = []
        self.val = None
        self.fval = True
        self.resetVals()

    def copy(self, fmol):
        """
        Copy the current fragment to a new one.

        :param fmol FragMol: the fragMol object this fragment belongs to.
        :return Fragment: the copied fragment.
        """
        frag = Fragment(self.dihe, fmol)
        frag.aids = self.aids[:]
        frag.pfrag = self.pfrag
        frag.nfrags = self.nfrags
        frag.vals = self.vals[:]
        frag.val = self.val
        frag.val = self.fval
        return frag

    def copyInit(self, fmol):
        """
        Copy the current initial fragment and all the fragments retrieved by it
        The connections between all new fragments are established as well.

        :param fmol FragMol: the fragMol object this initial fragment belongs to
        :return Fragment: the copied initial fragment.
        """
        ifrag = self.copy(fmol=fmol)
        all_nfrags = [ifrag]
        while (all_nfrags):
            frag = all_nfrags.pop()
            nfrags = [x.copy(fmol=fmol) for x in frag.nfrags]
            frag.nfrags = nfrags
            for nfrag in nfrags:
                nfrag.pfrag = frag
            all_nfrags += nfrags
        return ifrag

    def resetVals(self):
        """
        Reset the dihedral angle values and state.
        """
        self.fval, self.val = True, None
        self.vals = list(np.linspace(0, 360, 36, endpoint=False))

    def setFrags(self):
        """
        Set fragments by searching for rotatable bond path and adding them as
        next fragments.

        :return list of 'Fragment': newly added fragments with the first being itself
            (the atom ids of the current fragments changed)
        """
        dihes = self.getNewDihes()
        if not dihes:
            return []
        nfrags = [Fragment(x, self.fmol) for x in dihes]
        if self.dihe:
            nfrags = [self] + nfrags
        else:
            # Manually set self.dihe for the initial fragment
            self.dihe = dihes[0]
            self.setUp()
            nfrags[0] = self
        [x.setUp() for x in nfrags[1:]]
        for frag, nfrag in zip(nfrags[:-1], nfrags[1:]):
            frag.addFrag(nfrag)
        return nfrags

    def getNewDihes(self):
        """
        Get a list of dihedral angles that travel along the fragment rotatable
        bond to one fragment atom ids so that the path has the most rotatable
        bonds.

        NOTE: If the current dihedral angle of this fragment is not set, the
        dihedral angle list travel along the path with the most rotatable bonds.

        :return list of list: each sublist has four atom ids.
        """
        if self.dihe:
            pairs = zip([self.dihe[1]] * len(self.aids), self.aids)
        else:
            pairs = self.fmol.findPolymPair()

        dihes, num = [], 0
        for source, target in pairs:
            _, _, path = self.fmol.findPath(source=source, target=target)
            a_dihes = zip(path[:-3], path[1:-2], path[2:-1], path[3:])
            a_dihes = [x for x in a_dihes if self.fmol.isRotatable(x[1:-1])]
            if len(a_dihes) < num:
                continue
            num = len(a_dihes)
            dihes = a_dihes
        return dihes

    def setUp(self):
        """
        Set up the fragment.
        """
        self.aids = self.fmol.getSwingAtoms(*self.dihe)

    def addFrag(self, nfrag):
        """
        Add the next fragment to the current one's new fragments.

        :param nfrag 'Fragment': the next fragment to be added.
        """
        self.aids = sorted(set(self.aids).difference(nfrag.aids))
        self.nfrags.append(nfrag)

    def setDihedralDeg(self, val=None):
        """
        Set the dihedral angle of the current fragment. If val is not provided,
        randomly choose one from the candidates.

        :param val float: the dihedral angle value to be set.
        """
        if val is None:
            val = self.popVal()
        self.val = val
        Chem.rdMolTransforms.SetDihedralDeg(self.fmol.conf, *self.dihe,
                                            self.val)

    def getDihedralDeg(self):
        """
        Measure and return the dihedral angle value

        :return float: the dihedral angle degree
        """

        return Chem.rdMolTransforms.GetDihedralDeg(self.fmol.conf, *self.dihe)

    def popVal(self):
        """
        Randomly pop one val from dihedral value candidates.

        :return float: the picked dihedral value
        """
        val = np.random.choice(self.vals)
        self.vals.remove(val)
        self.fval = False
        return val

    def getPreAvailFrag(self):
        """
        Get the first previous fragment that has available dihedral candidates.

        :return 'Fragment': The previous fragment found.
        """
        frag = self.pfrag
        if frag is None:
            return None
        while (frag.pfrag and not frag.vals):
            frag = frag.pfrag
        if frag.pfrag is None and not frag.vals:
            # FIXME: Failed conformer search should try to reduce clash criteria
            return None
        return frag

    def getNxtFrags(self):
        """
        Get the next fragments that don't have full dihedral value candidates.

        :return list: list of next fragments
        """
        all_nfrags = []
        nfrags = [self]
        while (nfrags):
            nfrags = [y for x in nfrags for y in x.nfrags if not y.fval]
            all_nfrags += nfrags
        return all_nfrags

    def hasClashes(self):
        """
        Whether the atoms in current fragment has clashes with other atoms in
        the molecue that are set as existing atoms

        :return bool: clash exists or not.
        """
        return self.fmol.hasClashes(self.aids)

    def setConformer(self):
        """
        Try out dihedral angle values to avoid clashes.

        :return bool: True on the first no-clash conformer with respect to the
            current dihedral angle and atom ids.
        """
        self.fmol.setDcell()
        while (self.vals):
            self.setDihedralDeg()
            if not self.hasClashes():
                return True
        return False
