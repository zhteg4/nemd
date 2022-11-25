import traj
import logutils
import oplsua
import random
import numpy as np
import pandas as pd
from rdkit import Chem
import structutils

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):

    if logger is None:
        return
    logger.debug(msg)


class Fragment:

    def __repr__(self):
        return f"{self.dihe}: {self.atom_ids}"

    def __init__(self, dihe, fmol):
        self.dihe = dihe
        self.fmol = fmol
        self.atom_ids = []
        self.pfrag = None
        self.nfrags = []
        self.vals = []
        self.val = None
        self.fval = True
        self.resetVals()

    def resetVals(self):
        self.fval, self.val = True, None
        self.vals = list(np.linspace(0, 360, 36, endpoint=False))

    def getNewDihes(self):
        source = self.dihe[1] if self.dihe else None
        targets = self.atom_ids if self.atom_ids else [None]
        src, trgt, path, path_len = None, None, None, -1
        for target in targets:
            asrc, atrgt, apath = self.fmol.findLongPath(source=source,
                                                        target=target)
            if path_len >= len(apath):
                continue
            src, trgt, path, path_len = asrc, atrgt, apath, len(apath)
        dihes = list(zip(path[:-3], path[1:-2], path[2:-1], path[3:]))
        dihes = [x for x in dihes if self.fmol.isRotatable(x[1:-1])]
        return dihes

    def setFrags(self):
        dihes = self.getNewDihes()
        nfrags = [Fragment(x, self.fmol) for x in dihes]
        if self.dihe:
            nfrags = [self] + nfrags
        else:
            self.dihe = dihes[0]
            self.setUp()
            nfrags[0] = self
        [x.setUp() for x in nfrags[1:]]
        for frag, nfrag in zip(nfrags[:-1], nfrags[1:]):
            frag.addFrag(nfrag)
        return nfrags if dihes else []

    def addFrag(self, nfrag):
        self.atom_ids = sorted(set(self.atom_ids).difference(nfrag.atom_ids))
        self.nfrags.append(nfrag)

    def setUp(self):
        self.setSwingAtoms()
        self.setFragAtoms()

    def setSwingAtoms(self):
        self.swing_atom_ids = self.fmol.getSwingAtoms(*self.dihe)

    def setFragAtoms(self):
        self.atom_ids = self.swing_atom_ids[:]

    def popVal(self):
        val = random.choice(self.vals)
        self.vals.remove(val)
        self.fval = False
        return val

    def setDihedralDeg(self, val=None):
        if val is None:
            val = self.popVal()
        self.val = val
        Chem.rdMolTransforms.SetDihedralDeg(self.fmol.conf, *self.dihe,
                                            self.val)

    def getPreAvailFrag(self):
        frag, rfrags = self, []
        while (frag.pfrag and not frag.vals):
            frag = frag.pfrag
        if frag.pfrag is None and not frag.vals:
            raise ValueError('Conformer search failed.')
        return frag

    def getNxtFrags(self):
        all_nfrags = []
        nfrags = [self]
        while (nfrags):
            nfrags = [y for x in nfrags for y in x.nfrags if not y.fval]
            all_nfrags += nfrags
        return all_nfrags

    def hasClashes(self):
        return self.fmol.hasClashes(self.atom_ids)

    def setConformer(self):
        self.fmol.setDcell()
        while (self.vals):
            self.setDihedralDeg()
            if not self.hasClashes():
                return True
        return False


class FragMol:

    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    PATT = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')

    def __init__(self, mol, data_file=None):
        self.mol = mol
        self.data_file = data_file
        self.conf = self.mol.GetConformer(0)
        self.graph = structutils.getGraph(mol)
        self.rotatable_bonds = self.mol.GetSubstructMatches(self.PATT,
                                                            maxMatches=1000000)
        self.init_frag = None
        self.extg_aids = None

    def getSwingAtoms(self, *dihe):
        oxyz = self.conf.GetPositions()
        oval = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *dihe)
        Chem.rdMolTransforms.SetDihedralDeg(self.conf, *dihe, oval + 5)
        xyz = self.conf.GetPositions()
        changed = np.isclose(oxyz, xyz)
        Chem.rdMolTransforms.SetDihedralDeg(self.conf, *dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def isRotatable(self, bond):
        in_ring = self.mol.GetBondBetweenAtoms(*bond).IsInRing()
        single = tuple(sorted(bond)) in self.rotatable_bonds
        return not in_ring and single

    def findLongPath(self, source=None, target=None):
        return structutils.findLongPath(self.graph,
                                        source=source,
                                        target=target)

    def addNxtFrags(self):
        self.init_frag = Fragment([], self)
        to_be_fragmentized = self.init_frag.setFrags()
        while (to_be_fragmentized):
            frag = to_be_fragmentized.pop(0)
            nfrags = frag.setFrags()
            to_be_fragmentized += nfrags

    def setPreFrags(self):
        all_frags = self.fragments()
        for frag in all_frags:
            for nfrag in frag.nfrags:
                nfrag.pfrag = frag

    def fragments(self):
        all_frags = []
        nfrags = [self.init_frag]
        while (nfrags):
            all_frags += nfrags
            nfrags = [y for x in nfrags for y in x.nfrags]
        return all_frags

    def getNumFrags(self):
        pass

    def setInitAtomIds(self):
        frags = self.fragments()
        atom_ids = [y for x in frags for y in x.atom_ids]
        atom_ids_set = set(atom_ids)
        assert len(atom_ids) == len(atom_ids_set)
        self.extg_aids = set([
            x for x in range(self.mol.GetNumAtoms()) if x not in atom_ids_set
        ])

    def readData(self):
        self.data_reader = oplsua.DataFileReader(self.data_file)
        self.data_reader.run()
        self.data_reader.setClashParams(include14=True, scale=0.6)

    def setClashParams(self):
        self.max_clash_dist = max(
            [y for x in self.data_reader.radii.values() for y in x.values()])
        self.cell_rez = self.max_clash_dist
        self.cell_cut = self.max_clash_dist * 2

    def setCoords(self):
        for atom in self.data_reader.atoms.values():
            self.conf.SetAtomPosition(atom.id - 1, np.array(atom.xyz))

    def setFrm(self):
        box = np.array(
            [y for x in self.data_reader.box_dsp.values() for y in x])
        self.frm = pd.DataFrame(self.conf.GetPositions(),
                                index=range(1,
                                            self.conf.GetNumAtoms() + 1),
                                columns=['xu', 'yu', 'zu'])
        self.frm.attrs['box'] = box

    def hasClashes(self, atom_ids):
        self.frm.loc[:] = self.conf.GetPositions()
        frag_rows = [self.frm.iloc[x] for x in atom_ids]
        for row in frag_rows:
            clashes = self.dcell.getClashes(
                row,
                included=[x + 1 for x in self.extg_aids],
                radii=self.data_reader.radii,
                excluded=self.data_reader.excluded)
            if clashes:
                return True
        return False

    def setDcell(self):
        self.frm.loc[:] = self.conf.GetPositions()
        self.dcell = traj.DistanceCell(frm=self.frm,
                                       cut=self.cell_cut,
                                       resolution=self.cell_rez)
        self.dcell.setUp()

    def setConformer(self, seed=2022):
        random.seed(seed)
        frags = [self.init_frag]
        while (frags):
            frag = frags.pop(0)
            success = frag.setConformer()
            if success:
                self.extg_aids = self.extg_aids.union(frag.atom_ids)
                frags += frag.nfrags
                continue
            frag = frag.getPreAvailFrag()
            nxt_frags = frag.getNxtFrags()
            nxt_frags += [y for x in nxt_frags for y in x.nfrags]
            [x.resetVals() for x in nxt_frags]
            frags = [frag] + [x for x in frags if x not in nxt_frags]
            ratom_ids = frag.atom_ids[:]
            ratom_ids += [y for x in nxt_frags for y in x.atom_ids]
            self.extg_aids = self.extg_aids.difference(ratom_ids)
            log_debug(f"{len(self.extg_aids)}, {len(frag.vals)}: {frag}")

    def run(self):
        self.addNxtFrags()
        self.setPreFrags()
        self.setInitAtomIds()
        self.readData()
        self.setClashParams()
        self.setCoords()
        self.setFrm()
        self.setConformer(2022)
