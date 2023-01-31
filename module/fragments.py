import traj
import oplsua
import logutils
import itertools
import prop_names
import structutils
import conformerutils
import numpy as np
import pandas as pd
from rdkit import Chem

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):

    if logger is None:
        return
    logger.debug(msg)


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


class FragMixIn:

    def readData(self):
        """
        Read data  file and set clash parameters.
        """
        self.data_reader = oplsua.DataFileReader(self.data_file)
        self.data_reader.run()
        self.data_reader.setClashParams()

    def setDCellParams(self):
        """
        Set distance cell parameters.
        """

        self.max_clash_dist = self.data_reader.radii.max()
        self.cell_cut = self.max_clash_dist


class FragMol(FragMixIn):
    """
    Fragment molecule class to hold fragment information.
    """

    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    PATT = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
    IS_MONO = prop_names.IS_MONO
    MONO_ID = prop_names.MONO_ID
    POLYM_HT = prop_names.POLYM_HT

    def __init__(self, mol, data_file=None):
        """
        :param mol 'rdkit.Chem.rdchem.Mol': the molecule to fragmentize
        :param data_file str: filename path to get force field information
        """

        self.mol = mol
        self.data_file = data_file
        self.conf = self.mol.GetConformer(0)
        self.graph = structutils.getGraph(mol)
        self.rotatable_bonds = self.mol.GetSubstructMatches(self.PATT,
                                                            maxMatches=1000000)
        self.init_frag = None
        self.extg_aids = None
        self.frm = None

    def isRotatable(self, bond):
        """
        Whether the bond between the two atoms is rotatable.

        :param bond list or tuple of two ints: the atom ids of two bonded atoms
        :return bool: Whether the bond is rotatable.
        """

        in_ring = self.mol.GetBondBetweenAtoms(*bond).IsInRing()
        single = tuple(sorted(bond)) in self.rotatable_bonds
        return not in_ring and single

    def getSwingAtoms(self, *dihe):
        """
        Get the swing atoms when the dihedral angle changes.

        :param dihe list of four ints: the atom ids that form a dihedral angle
        :return list of ints: the swing atom ids when the dihedral angle changes.
        """

        oxyz = self.conf.GetPositions()
        oval = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *dihe)
        Chem.rdMolTransforms.SetDihedralDeg(self.conf, *dihe, oval + 5)
        xyz = self.conf.GetPositions()
        changed = np.isclose(oxyz, xyz)
        Chem.rdMolTransforms.SetDihedralDeg(self.conf, *dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def findPath(self, source=None, target=None):
        """
        Find the shortest path between source and target. If source and target
        are not provided, shortest paths between all pairs are computed and the
        long path is returned.

        :param source int: the atom id that serves as the source.
        :param target int: the atom id that serves as the target.
        :return list of ints: the atom ids that form the shortest path.
        """

        return structutils.findPath(self.graph, source=source, target=target)

    def findPolymPair(self):
        """
        If the molecule is built from momomers, the atom pairs from
        selected from the first and last monomers.

        :return list or iterator of int tuple: each tuple is an atom id pair
        """
        if not self.mol.HasProp(self.IS_MONO) or not self.mol.GetBoolProp(
                self.IS_MONO):
            return [(None, None)]

        ht_mono_ids = {
            x.GetProp(self.MONO_ID): []
            for x in self.mol.GetAtoms() if x.HasProp(self.POLYM_HT)
        }
        for atom in self.mol.GetAtoms():
            try:
                ht_mono_ids[atom.GetProp(self.MONO_ID)].append(atom.GetIdx())
            except KeyError:
                pass

        st_atoms = list(ht_mono_ids.values())
        sources = st_atoms[0]
        targets = [y for x in st_atoms[1:] for y in x]

        return itertools.product(sources, targets)

    def addNxtFrags(self):
        """
        Starting from the initial fragment, keep fragmentizing the current
        fragment and adding the newly generated ones to be fragmentized until
        no fragments can be further fragmentized.
        """
        self.init_frag = Fragment([], self)
        to_be_fragmentized = self.init_frag.setFrags()
        while (to_be_fragmentized):
            frag = to_be_fragmentized.pop(0)
            nfrags = frag.setFrags()
            to_be_fragmentized += nfrags
        if not self.init_frag.dihe:
            self.init_frag = None

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
        if self.init_frag is None:
            return all_frags
        nfrags = [self.init_frag]
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
            [x for x in range(self.mol.GetNumAtoms()) if x not in aids_set])

    def setGlobalAtomIds(self):
        """
        Set Global atom ids for each fragment and existing atoms.
        """
        self.gids = [x.GetAtomMapNum() for x in self.mol.GetAtoms()]
        self.extg_gids = set([
            self.mol.GetAtomWithIdx(x).GetAtomMapNum() for x in self.extg_aids
        ])
        for frag in self.fragments():
            frag.gids = [
                self.mol.GetAtomWithIdx(x).GetAtomMapNum() for x in frag.aids
            ]

    def setCoords(self):
        """
        Set conformer coordinates from data file.
        """
        for atom in self.data_reader.atoms.values():
            self.conf.SetAtomPosition(atom.id - 1, np.array(atom.xyz))

    def setFrm(self):
        """
        Set traj frame.
        """
        box = np.array(
            [y for x in self.data_reader.box_dsp.values() for y in x])
        xyz = self.conf.GetPositions()
        self.frm = traj.Frame(xyz=xyz, box=box)

    def setDcell(self):
        """
        Set distance cell.
        """
        self.frm.loc[:] = self.conf.GetPositions()
        self.dcell = traj.DistanceCell(frm=self.frm,
                                       cut=self.cell_cut,
                                       resolution='auto')
        self.dcell.setUp()

    def hasClashes(self, aids):
        """
        Whether the input atoms have clashes with the existing atoms.

        :param aids list of ints: list of atom ids
        :return bool: clashes exist or not.
        """

        self.frm.loc[:] = self.conf.GetPositions()
        frag_rows = [self.frm.iloc[x] for x in aids]
        for row in frag_rows:
            clashes = self.dcell.getClashes(
                row,
                included=[x + 1 for x in self.extg_aids],
                radii=self.data_reader.radii,
                excluded=self.data_reader.excluded)
            if clashes:
                return True
        return False

    def setConformer(self):
        """
        Set conformer coordinates without clashes.
        """
        log_debug(f"{self.getNumFrags()} fragments found.")
        frags = [self.init_frag]
        while (frags):
            frag = frags.pop(0)
            success = frag.setConformer()
            if success:
                self.extg_aids.update(frag.aids)
                frags += frag.nfrags
                continue
            # 1）Find the previous fragment with available dihedral candidates.
            frag = frag.getPreAvailFrag()
            # 2）Find the next fragments who have been placed into the cell.
            nxt_frags = frag.getNxtFrags()
            [x.resetVals() for x in nxt_frags]
            raids = [y for x in [frag] + nxt_frags for y in x.aids]
            self.extg_aids = self.extg_aids.difference(raids)
            # 3）Fragment after the next fragments were added to the growing
            # frags before this backmove step.
            nnxt_frags = [y for x in nxt_frags for y in x.nfrags]
            frags = [frag] + list(set(frags).difference(nnxt_frags))
            log_debug(f"{len(self.extg_aids)}, {len(frag.vals)}: {frag}")

    def setInitSize(self, buffer=1.):
        xyzs = np.array([self.conf.GetAtomPosition(x) for x in self.extg_aids])
        xyzs -= self.getInitCentroid()
        self.init_radius = np.linalg.norm(xyzs, axis=1).max() + buffer

    def getInitCentroid(self):
        return conformerutils.centroid(self.conf, aids=list(self.extg_aids))

    @property
    def molecule_id(self):
        return self.mol.GetIntProp('mol_id')

    def run(self):
        """
        Main method for fragmentation and conformer search.
        """
        self.addNxtFrags()
        self.setPreFrags()
        self.setInitAtomIds()
        self.readData()
        self.setDCellParams()
        self.setCoords()
        self.setFrm()
        self.setConformer()


class FragMols(FragMixIn):

    def __init__(self, mols, data_file=None, box=None, logger=logger):
        self.mols = mols
        self.data_file = data_file
        self.box = box
        self.logger = logger
        self.confs = None
        self.fmols = None
        self.failed_num = 0  # The failed attempts in growing molecules
        self.mol_num = None  # the last reported growing molecule number

    def log(self, msg, timestamp=False):
        if not self.logger:
            return
        logutils.log(self.logger, msg, timestamp=timestamp)

    def run(self):
        self.fragmentize()
        self.readData()
        self.setDCellParams()
        self.setCoords()
        self.setFrm()
        self.setConformer()

    def fragmentize(self):
        self.fmols = {i: FragMol(x) for i, x in self.mols.items()}
        for fmol in self.fmols.values():
            fmol.addNxtFrags()
            fmol.setPreFrags()
            fmol.setInitAtomIds()
            fmol.setGlobalAtomIds()
            fmol.setInitSize()
        total_frag_num = sum([x.getNumFrags() for x in self.fmols.values()])
        log_debug(f"{total_frag_num} fragments in total.")

    def setCoords(self):
        """
        Set conformer coordinates from data file.
        """

        for mol in self.mols.values():
            for atom in mol.GetAtoms():
                xyz = self.data_reader.atoms[atom.GetAtomMapNum()].xyz
                mol.GetConformer().SetAtomPosition(atom.GetIdx(),
                                                   np.array(xyz))
        self.confs = {i: x.GetConformer() for i, x in self.mols.items()}

    def setFrm(self):
        """
        Set traj frame.
        """
        xyz = np.array([x.xyz for x in self.data_reader.atoms.values()])
        self.frm = traj.Frame(xyz=xyz, box=self.box)

    def updateFrm(self):
        pos = [x.GetPositions() for x in self.confs.values()]
        self.frm.loc[:] = np.concatenate(pos, axis=0)

    def setDcell(self):
        """
        Set distance cell for neighbor atom and graph for voids searching.
        """
        self.updateFrm()
        self.dcell = traj.DistanceCell(frm=self.frm,
                                       cut=self.cell_cut,
                                       resolution=traj.DistanceCell.AUTO)
        self.dcell.setUp()
        self.dcell.setGraph()

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
        self.log(f'{finished_num} finished; {self.failed_num} failed.')
        return cur_mol_num

    def setConformer(self):
        """
        Set conformer coordinates without clashes.
        """

        frags = [x.init_frag for x in self.fmols.values()]
        self.setInitFrm(frags)
        self.setDcell()
        for frag in frags:
            self.placeInitFrag(frag)
        self.logInitFragsPlaced(frags)

        while frags:
            frag = frags.pop(0)
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
                    self.dcell.setGraph()
                    self.placeInitFrag(frags[0])
                    self.reportRelocation(frags[0])
            log_debug(f'{len(self.dcell.extg_gids)} atoms placed.')

    def logInitFragsPlaced(self, frags):
        """
        Log the initiator fragments status after the first placements.

        :param frags list of 'fragments.Fragment': the initiator of each frag
            has been placed into the cell.
        """

        self.log(f'{len(frags)} initiators have been placed into the cell.')
        if len(self.mols) == 1:
            return
        self.log(
            f'({self.getInitDists().min():.2f} as the minimum pair distance)')

    def getInitDists(self):
        """
        Calculate the initiator pair distances.

        :return array: each value is a distance between two initiators.
        """
        xyz = self.init_df.loc[:, self.init_df.columns != 'radius'].to_numpy()
        return traj.Frame(xyz=xyz, box=self.box).pairDists()

    def setInitFrm(self, frags):
        """
        Set the
        :param frags:
        :return:
        """
        data = np.array([[x.fmol.init_radius] + [np.inf, np.inf, np.inf]
                         for x in frags])
        index = [x.fmol.mol.GetIntProp('mol_id') for x in frags]
        self.init_df = pd.DataFrame(data=data,
                                    index=index,
                                    columns=['radius'] + traj.Frame.UXYZ)

    def placeInitFrag(self, frag):
        self.dcell.rmClashNodes()
        points = self.dcell.getVoids()
        for point in points:
            conf = frag.fmol.mol.GetConformer()
            aids = list(frag.fmol.extg_aids)
            centroid = np.array(conformerutils.centroid(conf, aids=aids))
            conformerutils.translation(conf, -centroid)
            conformerutils.rand_rotate(conf)
            conformerutils.translation(conf, point)
            self.frm.loc[frag.fmol.gids] = conf.GetPositions()
            gids = frag.fmol.extg_gids
            if self.hasClashes(gids):
                continue
            # Only update the distance cell after one molecule successful
            # placed into the cell as only inter-molecular clashes are
            # checked for packed cell.
            self.add(list(gids))
            self.init_df.loc[frag.fmol.molecule_id][1:] = point
            return
        raise ValueError(f'Failed to relocate the dead molecule.')

    def reportRelocation(self, frag):
        idists = self.getInitDists()
        dists = self.dcell.getDistsWithIds(frag.fmol.extg_gids)
        self.log(f"Relocate the initiator of "
                 f"{frag.fmol.mol.GetIntProp('mol_id')} "
                 f"(initiator: {idists.min():.2f}-{idists.max():.2f};"
                 f"close contact: {dists.min():.2f}) ")

    def backMove(self, frag, frags):
        # 1）Find the previous fragment with available dihedral candidates.
        pfrag = frag.getPreAvailFrag()
        found = bool(pfrag)
        frag = pfrag if found else frag.fmol.init_frag
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

    def hasClashes(self, gids):
        frag_rows = [self.frm.loc[x] for x in gids]
        for row in frag_rows:
            clashes = self.dcell.getClashes(row,
                                            included=self.dcell.extg_gids,
                                            radii=self.data_reader.radii,
                                            excluded=self.data_reader.excluded)
            if clashes:
                return True
        return False
