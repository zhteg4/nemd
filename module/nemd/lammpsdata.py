import io
import copy
import math
import struct

import scipy
import types
import base64
import itertools
import collections
import numpy as np
from scipy import constants

from nemd import oplsua
from nemd import symbols
from nemd import logutils
from nemd import lammpsin

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


class Struct(struct.Struct):
    ...


class LammpsDataBase(lammpsin.LammpsIn):

    LAMMPS_DESCRIPTION = 'LAMMPS Description # %s'

    METAL = 'metal'
    ATOMIC = 'atomic'

    ATOMS = 'atoms'
    ATOM_TYPES = 'atom types'

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    BOX_DSP = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    BUFFER = [4., 4., 4.] # yapf: disable

    MASSES = 'Masses'
    ATOM_ID = 'atom_id'
    TYPE_ID = oplsua.TYPE_ID

    def __init__(self, struct, *arg, ff=None, jobname='tmp', **kwarg):
        """
        :param struct 'Struct': structure with molecules and conformers
        :param ff 'oplsua.OplsParser': the force field information
        :param jobname str: jobname based on which out filenames are defined
        """
        # super(Struct, self).__init__(struct)
        super().__init__(jobname=jobname, *arg, **kwarg)
        self.struct = struct
        self.ff = ff
        self.jobname = jobname
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.nbr_charge = {}

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        charges = [
            self.ff.charges[x.GetIntProp(self.TYPE_ID)]
            for x in self.struct.atoms
        ]
        return any(charges)


class LammpsDataOne(LammpsDataBase):
    """
    Class to set bond, angle, dihedral, improper parameters, and other topology
    information.
    """
    RES_NUM = oplsua.RES_NUM
    IMPROPER_CENTER_SYMBOLS = symbols.CARBON + symbols.HYDROGEN

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.symbol_impropers = {}
        self.atm_types = {}
        self.bnd_types = {}
        self.ang_types = {}
        self.dihe_types = {}
        self.impr_types = {}
        self.rvrs_bonds = {}
        self.rvrs_angles = {}
        self.dihe_map = None

    def adjustCoords(self):
        """
        Adjust the coordinates based bond length etc.
        """
        self.setBonds()
        self.adjustBondLength()

    def run(self, adjust_coords=True):
        """
        Set charge, bond, angle, dihedral, improper, and other topology params.
        """
        self.balanceCharge()
        self.setBonds()
        self.adjustBondLength(adjust_coords)
        self.setAngles()
        self.setDihedrals()
        self.setImproperSymbols()
        self.setImpropers()
        self.removeAngles()

    def balanceCharge(self):
        """
        Balance the charge when residues are not neutral.
        """

        for mol_id, mol in self.struct.mols.items():
            # residual num: residual charge
            res_charge = collections.defaultdict(float)
            for atom in mol.GetAtoms():
                res_num = atom.GetIntProp(self.RES_NUM)
                type_id = atom.GetIntProp(self.TYPE_ID)
                res_charge[res_num] += self.ff.charges[type_id]

            res_snacharge = {x: 0 for x, y in res_charge.items() if y}
            res_atom = {}
            for bond in mol.GetBonds():
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
                    snatom_charge = abs(self.ff.charges[natom.GetIntProp(
                        self.TYPE_ID)])
                    if snatom_charge > res_snacharge[nres_num]:
                        res_atom[nres_num] = atom.GetIdx()
                        res_snacharge[nres_num] = snatom_charge
            nbr_charge = collections.defaultdict(float)
            for res, idx in res_atom.items():
                nbr_charge[idx] -= res_charge[res]
            self.nbr_charge[mol_id] = nbr_charge

    def setBonds(self):
        """
        Set bonding information.
        """
        bonds = [y for x in self.struct.molecules for y in x.GetBonds()]
        for bond_id, bond in enumerate(bonds, start=1):
            bonded = [bond.GetBeginAtom(), bond.GetEndAtom()]
            bond = self.ff.getMatchedBonds(bonded)[0]
            atom_id1 = bonded[0].GetAtomMapNum()
            atom_id2 = bonded[1].GetAtomMapNum()
            atom_ids = sorted([atom_id1, atom_id2])
            self.bonds[bond_id] = tuple([bond.id, *atom_ids])
            self.rvrs_bonds[tuple(atom_ids)] = bond.id

    def adjustBondLength(self, adjust_bond_legnth=True):
        """
        Adjust bond length according to the force field parameters.

        :param adjust_bond_legnth bool: adjust bond length if True.
        """
        if not adjust_bond_legnth:
            return

        for mol in self.struct.molecules:
            # Set the bond lengths of one conformer
            tpl = mol.GetConformer()
            for bond in mol.GetBonds():
                bonded = [bond.GetBeginAtom(), bond.GetEndAtom()]
                gids = set([x.GetAtomMapNum() for x in bonded])
                bond_type = self.rvrs_bonds[tuple(sorted(gids))]
                dist = self.ff.bonds[bond_type].dist
                tpl.setBondLength([x.GetIdx() for x in bonded], dist)
            # Update all conformers
            xyz = tpl.GetPositions()
            for conf in mol.GetConformers():
                conf.setPositions(xyz)

    def setAngles(self):
        """
        Set angle force field matches.
        """

        angs = [y for x in self.struct.atoms for y in self.ff.getAngleAtoms(x)]
        for angle_id, atoms in enumerate(angs, start=1):
            angle = self.ff.getMatchedAngles(atoms)[0]
            atom_ids = tuple(x.GetAtomMapNum() for x in atoms)
            self.angles[angle_id] = (angle.id, ) + atom_ids
            self.rvrs_angles[tuple(atom_ids)] = angle_id

    def setDihedrals(self):
        """
        Set the dihedral angles of the molecules.
        """

        dihe_atoms = [
            y for x in self.struct.molecules
            for y in self.getDihAtomsFromMol(x)
        ]
        for dihedral_id, atoms in enumerate(dihe_atoms, start=1):
            dihedral = self.ff.getMatchedDihedrals(atoms)[0]
            atom_ids = tuple([x.GetAtomMapNum() for x in atoms])
            self.dihedrals[dihedral_id] = (dihedral.id, ) + atom_ids

    def getDihAtomsFromMol(self, mol):
        """
        Get the dihedral atoms of this molecule.

        NOTE: Flipping the order the four dihedral atoms yields the same dihedral,
        and only one of them is returned.

        :param 'rdkit.Chem.rdchem.Mol': the molecule to get dihedral atoms.
        :return list of list: each sublist has four atom ids forming a dihedral angle.
        """
        atomss = [y for x in mol.GetAtoms() for y in self.getDihedralAtoms(x)]
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

    def setImproperSymbols(self):
        """
        Check and assert the current improper force field. These checks may be
        only good for this specific force field for even this specific file.
        """
        msg = "Impropers from the same symbols are of the same constants."
        # {1: 'CNCO', 2: 'CNCO', 3: 'CNCO' ...
        symbolss = {
            z:
            ''.join([str(self.ff.atoms[x.id3].conn)] + [
                self.ff.atoms[y].symbol for y in [x.id1, x.id2, x.id3, x.id4]
            ])
            for z, x in self.ff.impropers.items()
        }
        # {'CNCO': (10.5, 180.0, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9), ...
        symbol_impropers = {}
        for id, symbols in symbolss.items():
            improper = self.ff.impropers[id]
            if symbols not in symbol_impropers:
                symbol_impropers[symbols] = (
                    improper.ene,
                    improper.angle,
                    improper.n_parm,
                )
            assert symbol_impropers[symbols][:3] == (
                improper.ene,
                improper.angle,
                improper.n_parm,
            )
            symbol_impropers[symbols] += (improper.id, )
        log_debug(msg)

        # neighbors of CC(=O)C and CC(O)C have the same symbols
        msg = "Improper neighbor counts based on center conn and symbols are unique."
        # The third one is the center ('Improper Torsional Parameters' in prm)
        neighbors = [[x[0], x[3], x[1], x[2], x[4]]
                     for x in symbol_impropers.keys()]
        # The csmbls in getCountedSymbols is obtained from the following
        csmbls = sorted(set([y for x in neighbors for y in x[1:]]))  # CHNO
        counted = [self.countSymbols(x, csmbls=csmbls) for x in neighbors]
        assert len(symbol_impropers) == len(set(counted))
        log_debug(msg)
        self.symbol_impropers = {
            x: y[3:]
            for x, y in zip(counted, symbol_impropers.values())
        }

    @staticmethod
    def countSymbols(symbols, csmbls='CHNO'):
        """
        Count improper cluster symbols: the first is the center atom connectivity
        including implicit hydrogen atoms. The second is the center atom symbol,
        and the rest connects with the center.

        :param symbols list: the element symbols forming the improper cluster
            with first being the center
        :param csmbls str: all possible cluster symbols
        """
        # e.g., ['3', 'C', 'C', 'N', 'O']
        counted = [y + str(symbols[2:].count(y)) for y in csmbls]
        # e.g., '3CC1H0N1O1'
        return ''.join(symbols[:2] + counted)

    def setImpropers(self, csymbols=IMPROPER_CENTER_SYMBOLS):
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
        improper_id = 0
        for atom in self.struct.atoms:
            atom_symbol, neighbors = atom.GetSymbol(), atom.GetNeighbors()
            if atom_symbol not in csymbols or len(neighbors) != 3:
                continue
            if atom.GetSymbol() == symbols.NITROGEN and atom.GetHybridization(
            ) == Chem.rdchem.HybridizationType.SP3:
                continue
            # Sp2 carbon for planar, Sp3 with one H (CHR1R2R3) for chirality,
            # Sp2 N in Amino Acid
            improper_id += 1
            neighbor_symbols = [x.GetSymbol() for x in neighbors]
            counted = self.countSymbols(
                [str(oplsua.OplsParser.getAtomConnt(atom)), atom_symbol] +
                neighbor_symbols)
            improper_type_id = self.symbol_impropers[counted][0]
            # FIXME: see docstring for current investigation. (NO ACTIONS TAKEN)
            #  1) LAMMPS recommends the first to be the center, while the prm
            #  and literature order the third as the center.
            #  2) In addition, since improper has one non-connected edge,
            #  are the two non-edge atom selections important?
            #  3) Moreover, do we have to delete over constrained angle? If so,
            #  how about the one facing the non-connected edge?
            # My recommendation (not current implementation):
            # first plane: center + the two most heavy atom
            # second plane: the three non-center atoms
            # benefit: 1) O-C-O / O.O.R imposes symmetricity (RCOO)
            # 2) R-N-C / O.O.H exposes hydrogen out of plane vibration (RCNH)

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
            atoms = [neighbors[0], neighbors[1], atom, neighbors[2]]
            self.impropers[improper_id] = (improper_type_id, ) + tuple(
                x.GetAtomMapNum() for x in atoms)

    def printImpropers(self):
        """
        Print all the possible improper angles in the force field file.
        """
        for symb, improper_ids in self.symbol_impropers.items():
            print(f"{symb} {self.ff.impropers[improper_ids[0]]}")
            impropers = [self.ff.impropers[x] for x in improper_ids]
            for improper in impropers:
                ids = [improper.id1, improper.id2, improper.id3, improper.id4]
                print(f"{[self.ff.atoms[x].description for x in ids]}")

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
        ref: Atomic Forces for Geometry-Dependent Point Multipole and Gaussian
        Multipole Models
        """

        for idx, (itype, id1, id2, id3, id4) in self.impropers.items():
            for eids in itertools.combinations([id2, id1, id4], 2):
                angle_atom_ids = tuple([eids[0], id3, eids[1]])
                if angle_atom_ids not in self.rvrs_angles:
                    angle_atom_ids = angle_atom_ids[::-1]
                angle_type = self.angles[self.rvrs_angles[angle_atom_ids]][0]
                if np.isnan(self.ff.angles[angle_type].ene):
                    break
            self.angles.pop(self.rvrs_angles[angle_atom_ids])


class LammpsData(LammpsDataBase):
    ATOMS = LammpsDataBase.ATOMS
    BONDS = 'bonds'
    ANGLES = 'angles'
    DIHEDRALS = 'dihedrals'
    IMPROPERS = 'impropers'
    STRUCT_DSP = [ATOMS, BONDS, ANGLES, DIHEDRALS, IMPROPERS]

    ATOM_TYPES = LammpsDataBase.ATOM_TYPES
    BOND_TYPES = 'bond types'
    ANGLE_TYPES = 'angle types'
    DIHEDRAL_TYPES = 'dihedral types'
    IMPROPER_TYPES = 'improper types'
    TYPE_DSP = [
        ATOM_TYPES, BOND_TYPES, ANGLE_TYPES, DIHEDRAL_TYPES, IMPROPER_TYPES
    ]

    MASSES = LammpsDataBase.MASSES
    PAIR_COEFFS = 'Pair Coeffs'
    BOND_COEFFS = 'Bond Coeffs'
    ANGLE_COEFFS = 'Angle Coeffs'
    DIHEDRAL_COEFFS = 'Dihedral Coeffs'
    IMPROPER_COEFFS = 'Improper Coeffs'
    ATOMS_CAP = ATOMS.capitalize()
    BONDS_CAP = BONDS.capitalize()
    ANGLES_CAP = ANGLES.capitalize()
    DIHEDRALS_CAP = DIHEDRALS.capitalize()
    IMPROPERS_CAP = IMPROPERS.capitalize()

    MARKERS = [
        MASSES, PAIR_COEFFS, BOND_COEFFS, ANGLE_COEFFS, DIHEDRAL_COEFFS,
        IMPROPER_COEFFS, ATOMS_CAP, BONDS_CAP, ANGLES_CAP, DIHEDRALS_CAP,
        IMPROPERS_CAP
    ]

    def __init__(self,
                 struct,
                 *arg,
                 ff=None,
                 jobname='tmp',
                 concise=True,
                 box=None,
                 **kwarg):
        """
        :param struct Struct: struct object with moelcules and conformers.
        :param ff 'oplsua.OplsParser': the force field information
        :param jobname str: jobname based on which out filenames are defined
        :param concise bool: If False, all the atoms in the force field file
            shows up in the force field section of the data file. If True, only
            the present ones are writen into the data file.
        :param box list: the PBC limits (xlo, xhi, ylo, yhi, zlo, zhi)
        """
        super().__init__(struct, *arg, ff=ff, jobname=jobname, **kwarg)
        self.concise = concise
        self.box = box
        self.mol_dat = {}
        self.total_charge = 0.
        self.data_hdl = None
        self.density = None

    def writeRun(self, *arg, **kwarg):
        """
        Write command to further equilibrate the system with molecules
        information considered.
        """
        super().writeRun(*arg,
                         mols=self.struct.mols,
                         struct=self.struct,
                         **kwarg)

    def writeDumpModify(self):
        """
        Write dump modify commands so that dump command can write out element.
        """
        atoms = self.ff.atoms.values()
        if self.concise:
            atoms = [x for x in atoms if x.id in self.atm_types]
        smbs = ' '.join(map(str, [x.symbol for x in atoms]))
        self.in_fh.write(f"dump_modify 1 element {smbs}\n")

    def writeFixShake(self):
        """
        Write the fix shake so that the bonds and angles associated with hydrogen
        atoms keep constant.
        """
        fix_bonds = set()
        for btype, btype_concise in self.bnd_types.items():
            bond = self.ff.bonds[btype]
            atoms = [self.ff.atoms[x] for x in [bond.id1, bond.id2]]
            has_h = any(x.symbol == symbols.HYDROGEN for x in atoms)
            if has_h:
                bond_type = btype_concise if self.concise else btype
                fix_bonds.add(bond_type)

        fix_angles = set()
        for atype, atype_concise in self.ang_types.items():
            angle = self.ff.angles[atype]
            atoms = [
                self.ff.atoms[x] for x in [angle.id1, angle.id2, angle.id3]
            ]
            has_h = any(x.symbol == symbols.HYDROGEN for x in atoms)
            if has_h:
                angle_type = atype_concise if self.concise else atype
                fix_angles.add(angle_type)
        btype_ids = ' '.join(map(str, fix_bonds))
        atype_ids = ' '.join(map(str, fix_angles))
        if not any([btype_ids, atype_ids]):
            return
        self.in_fh.write(
            f'fix rigid all shake 0.0001 10 10000 b {btype_ids} a {atype_ids}\n'
        )

    def setOneMolData(self, adjust_coords=True):
        """
        Set one molecule for each molecule type.

        :param adjust_coords bool: whether adjust coordinates of the molecules.
            This only good for a small piece as clashes between non-bonded atoms
            may be introduced.
        """
        for mol_id, mol in self.struct.mols.items():
            struct = copy.copy(self.struct)
            struct.mols = {mol_id: mol}
            mol_dat = LammpsDataOne(struct, ff=self.ff, jobname=self.jobname)
            mol_dat.run(adjust_coords=adjust_coords)
            self.mol_dat[mol_id] = mol_dat

    def writeData(self, adjust_coords=True, nofile=False):
        """
        Write out LAMMPS data file.

        :param adjust_coords bool: whether adjust coordinates of the molecules.
            This only good for a small piece as clashes between non-bonded atoms
            may be introduced.
        :param nofile bool: return the string instead of writing to a file if True
        """

        with io.StringIO() if nofile else open(self.datafile,
                                               'w') as self.data_hdl:
            self.setOneMolData(adjust_coords=adjust_coords)
            self.setBADI()
            self.removeUnused()
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writePairCoeffs()
            self.writeBondCoeffs()
            self.writeAngleCoeffs()
            self.writeDihedralCoeffs()
            self.writeImproperCoeffs()
            self.writeAtoms()
            self.writeBonds()
            self.writeAngles()
            self.writeDihedrals()
            self.writeImpropers()
            return self.getContents() if nofile else None

    def setBADI(self):
        """
        Set bond, angle, dihedral, and improper for all conformers across
        all molecules.
        """
        bond_id, angle_id, dihedral_id, improper_id, atom_num = [0] * 5
        for tpl_id, tpl_dat in self.mol_dat.items():
            self.nbr_charge[tpl_id] = tpl_dat.nbr_charge[tpl_id]
            for _ in range(tpl_dat.struct.mols[tpl_id].GetNumConformers()):
                for id in tpl_dat.bonds.values():
                    bond_id += 1
                    bond = tuple([id[0]] + [x + atom_num for x in id[1:]])
                    self.bonds[bond_id] = bond
                for id in tpl_dat.angles.values():
                    angle_id += 1
                    angle = tuple([id[0]] + [x + atom_num for x in id[1:]])
                    self.angles[angle_id] = angle
                for id in tpl_dat.dihedrals.values():
                    dihedral_id += 1
                    dihedral = tuple([id[0]] + [x + atom_num for x in id[1:]])
                    self.dihedrals[dihedral_id] = dihedral
                for id in tpl_dat.impropers.values():
                    improper_id += 1
                    improper = tuple([id[0]] + [x + atom_num for x in id[1:]])
                    self.impropers[improper_id] = improper
                atom_num += tpl_dat.struct.mols[tpl_id].GetNumAtoms()

    def removeUnused(self):
        """
        Remove used force field information so that the data file is minimal.
        """
        if not self.concise:
            return

        atypes = sorted(
            set(x.GetIntProp(self.TYPE_ID) for x in self.struct.atoms))
        self.atm_types = {y: x for x, y in enumerate(atypes, start=1)}
        btypes = sorted(set(x[0] for x in self.bonds.values()))
        self.bnd_types = {y: x for x, y in enumerate(btypes, start=1)}
        antypes = sorted(set(x[0] for x in self.angles.values()))
        self.ang_types = {y: x for x, y in enumerate(antypes, start=1)}
        dtps = sorted(set(x[0] for x in self.dihedrals.values()))
        self.dihe_types = {y: x for x, y in enumerate(dtps, start=1)}
        itps = sorted(set(x[0] for x in self.impropers.values()))
        self.impr_types = {y: x for x, y in enumerate(itps, start=1)}

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom, bond,
        angle etc.
        """
        lmp_dsp = self.LAMMPS_DESCRIPTION % self.atom_style
        self.data_hdl.write(f"{lmp_dsp}\n\n")
        self.data_hdl.write(f"{self.struct.atom_total} {self.ATOMS}\n")
        self.data_hdl.write(f"{len(self.bonds)} {self.BONDS}\n")
        self.data_hdl.write(f"{len(self.angles)} {self.ANGLES}\n")
        self.data_hdl.write(f"{len(self.dihedrals)} {self.DIHEDRALS}\n")
        self.data_hdl.write(f"{len(self.impropers)} {self.IMPROPERS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        atom_num = len(self.atm_types) if self.concise else len(self.ff.atoms)
        self.data_hdl.write(f"{atom_num} {self.ATOM_TYPES}\n")
        bond_num = len(self.bnd_types) if self.concise else len(self.ff.bonds)
        self.data_hdl.write(f"{bond_num} {self.BOND_TYPES}\n")
        ang_num = len(self.ang_types) if self.concise else len(self.ff.angles)
        self.data_hdl.write(f"{ang_num} {self.ANGLE_TYPES}\n")
        dnum = len(self.dihe_types) if self.concise else len(self.ff.dihedrals)
        self.data_hdl.write(f"{dnum} {self.DIHEDRAL_TYPES}\n")
        inum = len(self.impr_types) if self.concise else len(self.ff.impropers)
        self.data_hdl.write(f"{inum} {self.IMPROPER_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        """
        Write box information.

        :param min_box list: minimum box size
        :param buffer list: buffer in three dimensions
        """

        xyzs = np.concatenate([
            y.GetPositions() for x in self.struct.mols.values()
            for y in x.GetConformers()
        ])
        ctr = xyzs.mean(axis=0)
        box_hf = self.getHalfBox(xyzs, min_box=min_box, buffer=buffer)
        box = [[x - y, x + y, z] for x, y, z in zip(ctr, box_hf, self.LO_HI)]
        if self.box is not None:
            boxes = zip(box, np.array(self.box).reshape(-1, 2))
            box = [[*x, symbols.POUND, *y] for x, y in boxes]
        for line in box:
            line = [f'{x:.2f}' if isinstance(x, float) else x for x in line]
            self.data_hdl.write(f"{' '.join(line)}\n")
        self.data_hdl.write("\n")
        # Calculate density as the revised box may alter the box size.
        weight = sum([
            self.ff.molecular_weight(x) * x.GetNumConformers()
            for x in self.struct.molecules
        ])
        edges = [
            x * 2 * scipy.constants.angstrom / scipy.constants.centi
            for x in box_hf
        ]
        self.density = weight / math.prod(edges) / scipy.constants.Avogadro

    def getHalfBox(self, xyzs, min_box=None, buffer=None):
        """
        Get the half box size based on interaction minimum, buffer, and structure
        span.

        :param xyzs 'numpy.ndarray': the xyz of the structure
        :param min_box list: minimum box size
        :param buffer list: the buffer in xyz dimensions (good for non-pbc)
        :return list of three floats: the xyz box limits.
        """
        if min_box is None:
            # PBC should be 2x larger than the cutoff, otherwise one particle
            # can interact with another particle within its cutoff twice: within
            # the box and across the PBC.
            cut_x2 = min([self.lj_cut, self.coul_cut]) * 2
            min_box = (cut_x2, cut_x2, cut_x2,)  # yapf: disable
        if buffer is None:
            buffer = self.BUFFER  # yapf: disable
        box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
        if self.box is not None:
            box = [(x - y) for x, y in zip(self.box[1::2], self.box[::2])]
        box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]
        if sum([x.GetNumConformers() for x in self.struct.mols.values()]) != 1:
            return box_hf
        # All-trans single molecule with internal tension runs into clashes
        # across PBCs and thus larger box is used.
        return [x * 1.2 for x in box_hf]

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.data_hdl.write(f"{self.MASSES}\n\n")
        for atom_id, atom in self.ff.atoms.items():
            if self.concise and atom_id not in self.atm_types:
                continue
            atm_id = self.atm_types[atom_id] if self.concise else atom_id
            dscrptn = f"{atom.description} {atom.symbol} {atom_id}" if self.concise else atom.description
            self.data_hdl.write(f"{atm_id} {atom.mass} # {dscrptn}\n")
        self.data_hdl.write(f"\n")

    def writePairCoeffs(self):
        """
        Write pair coefficients.
        """
        self.data_hdl.write(f"{self.PAIR_COEFFS}\n\n")
        for atom in self.ff.atoms.values():
            if self.concise and atom.id not in self.atm_types:
                continue
            vdw = self.ff.vdws[atom.id]
            atom_id = self.atm_types[atom.id] if self.concise else atom.id
            self.data_hdl.write(f"{atom_id} {vdw.ene:.4f} {vdw.dist:.4f}\n")
        self.data_hdl.write("\n")

    def writeBondCoeffs(self):
        """
        Write bond coefficients.
        """

        if not self.bnd_types:
            return

        self.data_hdl.write(f"{self.BOND_COEFFS}\n\n")
        for bond in self.ff.bonds.values():
            if self.concise and bond.id not in self.bnd_types:
                continue
            bond_id = self.bnd_types[bond.id] if self.concise else bond.id
            self.data_hdl.write(f"{bond_id}  {bond.ene} {bond.dist}\n")
        self.data_hdl.write("\n")

    def writeAngleCoeffs(self):
        """
        Write angle coefficients.
        """
        if not self.ang_types:
            return

        self.data_hdl.write(f"{self.ANGLE_COEFFS}\n\n")
        for angle in self.ff.angles.values():
            if self.concise and angle.id not in self.ang_types:
                continue
            angle_id = self.ang_types[angle.id] if self.concise else angle.id
            self.data_hdl.write(f"{angle_id} {angle.ene} {angle.angle}\n")
        self.data_hdl.write("\n")

    def writeDihedralCoeffs(self):
        """
        Write dihedral coefficients.
        """
        if not self.dihe_types:
            return

        self.data_hdl.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        for dihe in self.ff.dihedrals.values():
            if self.concise and dihe.id not in self.dihe_types:
                continue
            dihedral_id = self.dihe_types[dihe.id] if self.concise else dihe.id
            params = [0., 0., 0., 0.]
            # LAMMPS: K1, K2, K3, K4 in 0.5*K1[1+cos(x)] + 0.5*K2[1-cos(2x)]...
            # OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            for ene_ang_n in dihe.constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if params[ene_ang_n.n_parm] and ((ene_ang_n.angle == 180.) ^
                                                 (not ene_ang_n.n_parm % 2)):
                    params[ene_ang_n.n_parm] *= -1
            self.data_hdl.write(
                f"{dihedral_id}  {' '.join(map(str, params))}\n")
        self.data_hdl.write("\n")

    def writeImproperCoeffs(self):
        """
        Write improper coefficients.
        """
        if not self.impr_types:
            return

        self.data_hdl.write(f"{self.IMPROPER_COEFFS}\n\n")
        for impr in self.ff.impropers.values():
            if self.concise and impr.id not in self.impr_types:
                continue
            improper_id = self.impr_types[impr.id] if self.concise else impr.id
            # LAMMPS: K in K[1+d*cos(nx)] vs OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            sign = 1 if impr.angle == 0. else -1
            self.data_hdl.write(
                f"{improper_id} {impr.ene} {sign} {impr.n_parm}\n")
        self.data_hdl.write("\n")

    def writeAtoms(self, fmt='%i %i %i %.4f %.3f %.3f %.3f'):
        """
        Write atom coefficients.

        :param fmt str: the format of atom line in LAMMPS data file.
        """

        self.data_hdl.write(f"{self.ATOMS.capitalize()}\n\n")
        pre_atoms = 0
        for tpl_id, mol in self.struct.mols.items():
            data = np.zeros((mol.GetNumAtoms(), 7))
            data[:, 0] = [x.GetAtomMapNum() for x in mol.GetAtoms()]
            type_ids = [x.GetIntProp(self.TYPE_ID) for x in mol.GetAtoms()]
            data[:, 2] = [
                self.atm_types[x] if self.concise else x for x in type_ids
            ]
            charges = [
                self.nbr_charge[tpl_id][x.GetIdx()] for x in mol.GetAtoms()
            ]
            data[:, 3] = [
                x + self.ff.charges[y] for x, y in zip(charges, type_ids)
            ]
            data[:, 0] += pre_atoms
            for conformer in mol.GetConformers():
                data[:, 1] = conformer.gid
                data[:, 4:] = conformer.GetPositions()
                np.savetxt(self.data_hdl, data, fmt=fmt)
                # Increment atom ids by atom number in this conformer so that
                # the next writing starts from the atoms in previous conformers
                data[:, 0] += mol.GetNumAtoms()
                self.total_charge += data[:, 3].sum()
            # Atom ids in starts from atom ids in previous template molecules
            pre_atoms += mol.GetNumAtoms() * mol.GetNumConformers()
        self.data_hdl.write(f"\n")

    def writeBonds(self):
        """
        Write bond coefficients.
        """

        if not self.bonds:
            return

        self.data_hdl.write(f"{self.BONDS.capitalize()}\n\n")
        for bond_id, (bond_type, id1, id2) in self.bonds.items():
            bond_type = self.bnd_types[bond_type] if self.concise else bond_type
            self.data_hdl.write(f"{bond_id} {bond_type} {id1} {id2}\n")
        self.data_hdl.write(f"\n")

    def writeAngles(self):
        """
        Write angle coefficients.
        """
        if not self.angles:
            return
        self.data_hdl.write(f"{self.ANGLES.capitalize()}\n\n")
        # Some angles may be filtered out by improper
        for angle_id, value in enumerate(self.angles.items(), start=1):
            _, (type_id, id1, id2, id3) = value
            angle_type = self.ang_types[type_id] if self.concise else type_id
            self.data_hdl.write(f"{angle_id} {angle_type} {id1} {id2} {id3}\n")
        self.data_hdl.write(f"\n")

    def writeDihedrals(self):
        """
        Write dihedral coefficients.
        """
        if not self.dihedrals:
            return

        self.data_hdl.write(f"{self.DIHEDRALS.capitalize()}\n\n")
        for dihe_id, (type_id, id1, id2, id3, id4) in self.dihedrals.items():
            type_id = self.dihe_types[type_id] if self.concise else type_id
            self.data_hdl.write(
                f"{dihe_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_hdl.write(f"\n")

    def writeImpropers(self):
        """
        Write improper coefficients.
        """
        if not self.impropers:
            return

        self.data_hdl.write(f"{self.IMPROPERS.capitalize()}\n\n")
        for improper_id, (type_id, id1, id2, id3,
                          id4) in self.impropers.items():
            type_id = self.impr_types[type_id] if self.concise else type_id
            self.data_hdl.write(
                f"{improper_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_hdl.write(f"\n")

    def getContents(self):
        """
        Return datafile contents in base64 encoding.

        :return `bytes`: the contents of the data file in base64 encoding.
        """
        self.data_hdl.seek(0)
        contents = base64.b64encode(self.data_hdl.read().encode("utf-8"))
        return b','.join([b'lammps_datafile', contents])


class DataFileReader(LammpsData):
    """
    LAMMPS Data file reader
    """

    SCALE = 0.45

    def __init__(self, data_file=None, min_dist=1.4, contents=None):
        """
        :param data_file str: data file with path
        :param min_dist: the minimum distance as clash (some h-bond has zero vdw
            params and the water O..H hydrogen bond is above 1.4)
        :param contents `bytes`: parse the contents if data_file not provided.
        """
        self.data_file = data_file
        self.min_dist = min_dist
        self.contents = contents
        self.lines = None
        self.masses = {}
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.vdws = {}
        self.radii = None
        self.mols = {}
        self.excluded = collections.defaultdict(set)

    def run(self):
        """
        Main method to read and parse the data file.
        """
        self.read()
        self.indexLines()
        self.setDescription()
        self.setMasses()
        self.setPairCoeffs()
        self.setAtoms()
        self.setBonds()
        self.setAngles()
        self.setDihedrals()
        self.setImpropers()
        self.setMols()

    @property
    def molecular_weight(self):
        """
        The total molecular weight over all atoms.

        :return float: the total weight.
        """
        type_ids = [x.type_id for x in self.atom]
        return sum(self.masses[x].mass for x in type_ids)

    mw = molecular_weight

    def setMinimumDist(self):
        for id in self.vdws.keys():
            if self.vdws[id].dist < self.min_dist:
                self.vdws[id].dist = self.min_dist

    def read(self):
        """
        Read the data file and index lines by section marker.
        """
        if self.data_file:
            with open(self.data_file, 'r') as df_fh:
                self.lines = df_fh.readlines()
        else:
            content_type, content_string = self.contents.split(b',')
            decoded = base64.b64decode(content_string)
            self.lines = decoded.decode("utf-8").splitlines()

    def indexLines(self):
        self.mk_idxes = {
            x: i
            for i, l in enumerate(self.lines)
            for x in self.MARKERS if l.startswith(x)
        }

    def setDescription(self):
        """
        Parse the description section for topo counts, type counts, and box size
        """
        dsp_eidx = min(self.mk_idxes.values())
        # {'atoms': 1620, 'bonds': 1593, 'angles': 1566, 'dihedrals': 2511}
        self.struct_dsp = {
            y: int(self.lines[x].split(y)[0])
            for x in range(dsp_eidx)
            for y in self.STRUCT_DSP if y in self.lines[x]
        }
        # {'atom types': 7, 'bond types': 6, 'angle types': 5}
        self.dype_dsp = {
            y: int(self.lines[x].split(y)[0])
            for x in range(dsp_eidx)
            for y in self.TYPE_DSP if y in self.lines[x]
        }
        # {'xlo xhi': [-7.12, 35.44], 'ylo yhi': [-7.53, 34.26], 'zlo zhi': ..}
        self.box_dsp = {
            y: [float(z) for z in self.lines[x].split(y)[0].split()]
            for x in range(dsp_eidx)
            for y in self.BOX_DSP if y in self.lines[x]
        }

    def setMasses(self):
        """
        Parse the mass section for masses and elements.
        """
        sidx = self.mk_idxes[self.MASSES] + 2
        for id, lid in enumerate(
                range(sidx, sidx + self.dype_dsp[self.ATOM_TYPES]), 1):
            splitted = self.lines[lid].split()
            id, mass, ele = splitted[0], splitted[1], splitted[-2]
            self.masses[int(id)] = types.SimpleNamespace(id=int(id),
                                                         mass=float(mass),
                                                         ele=ele)

    def getBox(self):
        """
        Get the box.

        :return list of float: xlo, xhi, ylo, yhi, zlo, zhi
        """
        return [y for x in self.box_dsp.values() for y in x]

    def getBoxEdges(self):
        """
        Get the edges of the box.

        :return list of list: each sublist contains two points describing one
            edge.
        """
        box = self.getBox()
        return self.getEdgesFromList(box)

    @staticmethod
    def getEdgesFromList(lo_hi):
        """
        Get the edges from point list of low and high points.

        :param lo_hi list of float: xlo, xhi, ylo, yhi, zlo, zhi
        :return list of list: each sublist contains two points describing one
            edge.
        """
        lo_hi = [lo_hi[i * 2:i * 2 + 2] for i in range(3)]
        los = [lh[0] for lh in lo_hi]
        lo_edges = [[los[:], los[:]] for _ in range(3)]
        for index, (lo, hi) in enumerate(lo_hi):
            lo_edges[index][1][index] = hi
        his = [lh[1] for lh in lo_hi]
        hi_edges = [[his[:], his[:]] for _ in range(3)]
        for index, (lo, hi) in enumerate(lo_hi):
            hi_edges[index][1][index] = lo
        spnts = collections.deque([x[1] for x in lo_edges])
        epnts = collections.deque([x[1] for x in hi_edges])
        epnts.rotate(1)
        oedges = [[x, y] for x, y in zip(spnts, epnts)]
        epnts.rotate(1)
        oedges += [[x, y] for x, y in zip(spnts, epnts)]
        return lo_edges + hi_edges + oedges

    def setAtoms(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        sidx = self.mk_idxes[self.ATOMS_CAP] + 2
        for lid in range(sidx, sidx + self.struct_dsp[self.ATOMS]):
            id, mol_id, type_id, charge, x, y, z = self.lines[lid].split()[:7]
            self.atoms[int(id)] = types.SimpleNamespace(
                id=int(id),
                mol_id=int(mol_id),
                type_id=int(type_id),
                xyz=(float(x), float(y), float(z)),
                ele=self.masses[int(type_id)].ele)

    @property
    def atom(self):
        """
        Handy way to get all atoms.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return (x for x in self.atoms.values())

    @property
    def atom_num(self):
        """
        Handy way to get all atoms.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return len(self.atoms)

    @property
    def molecule(self):
        """
        Handy way to get all molecules.

        :return list of list: each sublist contains one int as atom id
        """

        return super().molecule

    def setMols(self):
        """
        Group atoms into molecules by molecule ids.
        """
        mols = collections.defaultdict(list)
        for atom in self.atoms.values():
            try:
                mols[atom.mol_id].append(atom.id)
            except AttributeError:
                # atomic style has no molecule ids
                return
        self.mols = dict(mols)

    def setBonds(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        try:
            sidx = self.mk_idxes[self.BONDS_CAP] + 2
        except KeyError:
            return

        for lid in range(sidx, sidx + self.struct_dsp[self.BONDS]):
            id, type_id, id1, id2 = self.lines[lid].split()
            self.bonds[int(id)] = types.SimpleNamespace(id=int(id),
                                                        type_id=int(type_id),
                                                        id1=int(id1),
                                                        id2=int(id2))

    def setAngles(self):
        """
        Parse the angle section for angle id and constructing atoms.
        """
        try:
            sidx = self.mk_idxes[self.ANGLES_CAP] + 2
        except KeyError:
            return
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.ANGLES]), 1):

            id, type_id, id1, id2, id3 = self.lines[lid].split()[:5]
            self.angles[int(id)] = types.SimpleNamespace(id=int(id),
                                                         type_id=int(type_id),
                                                         id1=int(id1),
                                                         id2=int(id2),
                                                         id3=int(id3))

    def setDihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.
        """
        try:
            sidx = self.mk_idxes[self.DIHEDRALS_CAP] + 2
        except KeyError:
            return
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.DIHEDRALS]), 1):
            id, type_id, id1, id2, id3, id4 = self.lines[lid].split()[:6]
            self.dihedrals[int(id)] = types.SimpleNamespace(
                id=int(id),
                type_id=int(type_id),
                id1=int(id1),
                id2=int(id2),
                id3=int(id3),
                id4=int(id4))

    def setImpropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.
        """
        try:
            sidx = self.mk_idxes[self.IMPROPERS_CAP] + 2
        except KeyError:
            return
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.IMPROPERS]), 1):
            id, type_id, id1, id2, id3, id4 = self.lines[lid].split()[:6]
            self.impropers[int(id)] = types.SimpleNamespace(
                id=int(id),
                type_id=int(type_id),
                id1=int(id1),
                id2=int(id2),
                id3=int(id3),
                id4=int(id4))

    def setClashParams(self, include14=False, scale=SCALE):
        """
        Set clash check related parameters including pair radii and exclusion.

        :param include14 bool: whether to include atom separated by 2 bonds for
            clash check.
        :param scale float: the scale param on vdw radius in clash check.
        """
        self.setClashExclusion(include14=not include14)
        self.setPairCoeffs()
        self.setVdwRadius(scale=scale)

    def setClashExclusion(self, include14=True):
        """
        Bonded atoms and atoms in angles are in the exclusion. If include14=True,
        the dihedral angles are in the exclusion as well.

        :param include14 bool: If True, 1-4 interaction in a dihedral angle count
            as exclusion.
        """
        pairs = set((x.id1, x.id2) for x in self.bonds.values())
        pairs = pairs.union((x.id1, x.id3) for x in self.angles.values())
        pairs = pairs.union([
            y for x in self.impropers.values()
            for y in itertools.combinations([x.id1, x.id2, x.id3, x.id4], 2)
        ])
        if include14:
            pairs = pairs.union(
                (x.id1, x.id4) for x in self.dihedrals.values())
        for id1, id2 in pairs:
            self.excluded[id1].add(id2)
            self.excluded[id2].add(id1)

    def setPairCoeffs(self):
        """
        Paser the pair coefficient section.
        """
        if self.PAIR_COEFFS not in self.mk_idxes:
            return
        sidx = self.mk_idxes[self.PAIR_COEFFS] + 2
        for lid in range(sidx, sidx + self.dype_dsp[self.ATOM_TYPES]):
            id, ene, dist = self.lines[lid].split()
            self.vdws[int(id)] = types.SimpleNamespace(id=int(id),
                                                       dist=float(dist),
                                                       ene=float(ene))

    def setVdwRadius(self, mix=LammpsData.GEOMETRIC, scale=1.):
        """
        Set the vdw radius based on the mixing rule and vdw radii.

        :param mix str: the mixing rules, including GEOMETRIC, ARITHMETIC, and
            SIXTHPOWER
        :param scale float: scale the vdw radius by this factor

        NOTE: the scaled radii here are more like diameters (or distance)
            between two sites.
        """
        if mix == LammpsData.GEOMETRIC:
            # LammpsData.GEOMETRIC is optimized for speed and is supported
            atom_types = sorted(set([x.type_id for x in self.atoms.values()]))
            radii = [0] + [self.vdws[x].dist for x in atom_types]
            radii = np.full((len(radii), len(radii)), radii, dtype='float16')
            radii[:, 0] = radii[0, :]
            radii *= radii.transpose()
            radii = np.sqrt(radii)
            radii *= pow(2, 1 / 6) * scale
            radii[radii < self.min_dist] = self.min_dist
            id_map = {x.id: x.type_id for x in self.atoms.values()}
            self.radii = Radius(radii, id_map=id_map)
            return

        radii = collections.defaultdict(dict)
        for id1, vdw1 in self.vdws.items():
            for id2, vdw2 in self.vdws.items():
                if mix == self.GEOMETRIC:
                    dist = pow(vdw1.dist * vdw2.dist, 0.5)
                elif mix == self.ARITHMETIC:
                    dist = (vdw1.dist + vdw2.dist) / 2
                elif mix == self.SIXTHPOWER:
                    dist = (pow(vdw1.dist, 6) + pow(vdw2.dist, 6)) / 2
                    dist = pow(dist, 1 / 6)
                dist *= pow(2, 1 / 6) * scale
                if dist < self.min_dist:
                    dist = self.min_dist
                radii[id1][id2] = round(dist, 4)

        self.radii = collections.defaultdict(dict)
        for atom1 in self.atoms.values():
            for atom2 in self.atoms.values():
                self.radii[atom1.id][atom2.id] = radii[atom1.type_id][
                    atom2.type_id]
        self.radii = dict(self.radii)

    def getMolXYZ(self, id):
        """
        Get the xyz coordinates of a molecule.

        :param id int: the molecule id.
        :return np.ndarray: the xyz coordinates of the molecule.
        """

        return np.array([self.atoms[x].xyz for x in self.mols[id]])


class Radius(np.ndarray):
    """
    Class to get vdw radius from atom id pair.
    """

    def __new__(cls, input_array, *args, id_map=None, **kwargs):
        """
        :param input_array np.ndarray: the radius array with type id as row index
        :param id_map dict: map atom id to type id
        """
        obj = np.asarray(input_array).view(cls)
        obj.id_map = id_map
        return obj

    def getRadius(self, aid1, aid2):
        """
        Get the radius between atoms from two global ids.

        :param aid1 int: one global atom id from the pair.
        :param aid2 int: the other global atom id from the pair.
        :return float: the vdw radius between the pair.
        """
        return self[self.id_map[aid1], self.id_map[aid2]]

    def setRadius(self, aid1, aid2, val):
        """
        Get the radius between atoms from two global ids.

        :param aid1 int: one global atom id from the pair.
        :param aid2 int: the other global atom id from the pair.
        :val float: the vdw radius between the pair to be set.
        """
        self[self.id_map[aid1], self.id_map[aid2]] = val
        self[self.id_map[aid2], self.id_map[aid1]] = val
