import io
import math
import scipy
import types
import base64
import itertools
import collections
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import constants

from nemd import oplsua
from nemd import symbols
from nemd import lammpsin
from nemd import structure
from nemd import constants as nconstant

ATOM_ID = 'atom_id'
MOL_ID = 'mol_id'
TYPE_ID = 'type_id'
CHARGE = 'charge'
XU = symbols.XU
YU = symbols.YU
ZU = symbols.ZU
XYZU = symbols.XYZU


class Conformer(structure.Conformer):

    TYPE_ID = oplsua.TYPE_ID

    @property
    def atoms(self):
        """
        Return atom information in the format of numpy array.

        :return `pandas.core.frame.DataFrame`: information such as atom global
            ids, molecule ids, atom type ids, charges, coordinates.
        """
        if not self.HasOwningMol():
            return
        mol = self.GetOwningMol()
        data = dict(mol_id=np.array([self.gid] * mol.GetNumAtoms()))
        data[TYPE_ID] = [x.GetIntProp(self.TYPE_ID) for x in mol.GetAtoms()]
        fchrg = [mol.ff.charges[x] for x in data[TYPE_ID]]
        aids = [x.GetIdx() for x in mol.GetAtoms()]
        nchrg = [mol.nbr_charge[x] for x in aids]
        data[CHARGE] = np.array([sum(x) for x in zip(fchrg, nchrg)])
        for dim, vals in zip(symbols.XYZU, self.GetPositions().transpose()):
            data[dim] = vals
        index = pd.Index(self.id_map[aids], name=ATOM_ID)
        return pd.DataFrame(data, index=index)

    @property
    def bonds(self):
        """
        Return bond information in the format of numpy array.

        :return 'numpy.ndarray': information such as bond ids and bonded atom
            ids.
        """
        if not self.HasOwningMol():
            return
        bonds = np.array(self.GetOwningMol().bonds)
        if bonds.any():
            bonds[:, 1:] = self.id_map[bonds[:, 1:]]
        return bonds

    @property
    def angles(self):
        """
        Return angle information in the format of numpy array.

        :return 'numpy.ndarray': information such as angle ids and connected
            atom ids.
        """
        if not self.HasOwningMol():
            return
        angles = np.array(self.GetOwningMol().angles)
        if angles.any():
            angles[:, 1:] = self.id_map[angles[:, 1:]]
        return angles

    @property
    def dihedrals(self):
        """
        Return dihedral angle information in the format of numpy array.

        :return 'numpy.ndarray': information such as dihedral ids and connected
            atom ids.
        """
        if not self.HasOwningMol():
            return
        dihes = np.array(self.GetOwningMol().dihedrals)
        if dihes.any():
            dihes[:, 1:] = self.id_map[dihes[:, 1:]]
        return dihes

    @property
    def impropers(self):
        """
        Return improper angle information in the format of numpy array.

        :return 'numpy.ndarray': information such as improper ids and connected
            atom ids.
        """
        if not self.HasOwningMol():
            return
        imprps = np.array(self.GetOwningMol().impropers)
        if imprps.any():
            imprps[:, 1:] = self.id_map[imprps[:, 1:]]
        return imprps


class Mol(structure.Mol):

    ConfClass = Conformer
    RES_NUM = oplsua.RES_NUM
    TYPE_ID = oplsua.TYPE_ID
    IMPROPER_CENTER_SYMBOLS = symbols.CARBON + symbols.HYDROGEN

    def __init__(self, *args, ff=None, **kwargs):
        """
        :param ff 'Parser': the force field class.
        """
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.symbol_impropers = {}
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.nbr_charge = collections.defaultdict(float)
        self.rvrs_bonds = {}
        self.rvrs_angles = {}
        self.fbonds = set()
        self.fangles = set()
        if self.ff is None and self.struct and hasattr(self.struct, 'ff'):
            self.ff = self.struct.ff
        if self.ff is None:
            self.ff = oplsua.get_parser()
        if self.delay:
            return
        self.setTopo()

    def setTopo(self):
        """
        Set charge, bond, angle, dihedral, improper, and other topology params.
        """
        self.typeAtoms()
        self.balanceCharge()
        self.setBonds()
        self.setAngles()
        self.setDihedrals()
        self.setImpropers()
        self.removeAngles()
        self.setFixGeom()

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
            type_id = atom.GetIntProp(self.TYPE_ID)
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
                snatom_charge = abs(self.ff.charges[natom.GetIntProp(
                    self.TYPE_ID)])
                if snatom_charge > res_snacharge[nres_num]:
                    res_atom[nres_num] = atom.GetIdx()
                    res_snacharge[nres_num] = snatom_charge

        for res, idx in res_atom.items():
            self.nbr_charge[idx] -= res_charge[res]

    def setBonds(self):
        """
        Set bonding information.
        """
        for bond_id, bond in enumerate(self.GetBonds()):
            bonded = [bond.GetBeginAtom(), bond.GetEndAtom()]
            bond = self.ff.getMatchedBonds(bonded)[0]
            aids = sorted([bonded[0].GetIdx(), bonded[1].GetIdx()])
            self.bonds.append(tuple([bond.id, *aids]))
            self.rvrs_bonds[tuple(aids)] = bond.id

    def setAngles(self):
        """
        Set angle force field matches.
        """
        angles = [y for x in self.GetAtoms() for y in self.ff.getAngleAtoms(x)]
        for angle_id, atoms in enumerate(angles):
            angle = self.ff.getMatchedAngles(atoms)[0]
            aids = tuple(x.GetIdx() for x in atoms)
            self.angles.append((angle.id, ) + aids)
            self.rvrs_angles[aids] = angle_id

    def setDihedrals(self):
        """
        Set the dihedral angles of the molecules.
        """
        for atoms in self.getDihAtoms():
            dihedral = self.ff.getMatchedDihedrals(atoms)[0]
            aids = tuple([x.GetIdx() for x in atoms])
            self.dihedrals.append((dihedral.id, ) + aids)

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
        for atom in self.GetAtoms():
            atom_symbol, neighbors = atom.GetSymbol(), atom.GetNeighbors()
            if atom_symbol not in csymbols or len(neighbors) != 3:
                continue
            if atom.GetSymbol() == symbols.NITROGEN and atom.GetHybridization(
            ) == Chem.rdchem.HybridizationType.SP3:
                continue
            # Sp2 carbon for planar, Sp3 with one H (CHR1R2R3) for chirality,
            # Sp2 N in Amino Acid
            neighbor_symbols = [x.GetSymbol() for x in neighbors]
            counted = self.ff.countSymbols(
                [str(oplsua.Parser.getAtomConnt(atom)), atom_symbol] +
                neighbor_symbols)
            improper_type_id = self.ff.improper_symbols[counted][0]
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
            improper = (improper_type_id, ) + tuple(x.GetIdx() for x in atoms)
            self.impropers.append(improper)

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

        to_remove = []
        for itype, id1, id2, id3, id4 in self.impropers:
            for eids in itertools.combinations([id2, id1, id4], 2):
                angle_atom_ids = tuple([eids[0], id3, eids[1]])
                if angle_atom_ids not in self.rvrs_angles:
                    angle_atom_ids = angle_atom_ids[::-1]
                index = self.rvrs_angles[angle_atom_ids]
                angle = self.angles[index]
                if np.isnan(self.ff.angles[angle[0]].ene):
                    break
            to_remove.append(index)
        for index in sorted(to_remove, reverse=True):
            self.angles.pop(index)

    def setFixGeom(self):
        """
        The lengths or angle values of these geometries remain unchanged during
        simulation.
        """
        bonds = [self.ff.bonds[x[0]] for x in self.bonds]
        self.fbonds = set([x[0] for x in bonds if x.has_h])
        angles = [self.ff.angles[x[0]] for x in self.angles]
        self.fangles = set([x[0] for x in angles if x.has_h])

    @property
    def bond_total(self):
        """
        Total number of bonds in the molecule.

        :return int: number of bonds across conformers.
        """
        return len(self.bonds) * self.GetNumConformers()

    @property
    def angle_total(self):
        """
        Total number of angles in the molecule.

        :return int: number of angles across conformers.
        """
        return len(self.angles) * self.GetNumConformers()

    @property
    def dihedral_total(self):
        """
        Total number of dihedral angles in the molecule.

        :return int: number of dihedral in across conformers.
        """
        return len(self.dihedrals) * self.GetNumConformers()

    @property
    def improper_total(self):
        """
        Total number of improper angles in the structure.

        :return int: number of improper angles across conformers.
        """
        return len(self.impropers) * self.GetNumConformers()

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight


class Base(lammpsin.In):

    LAMMPS_DESCRIPTION = 'LAMMPS Description # %s'

    ATOM_TYPES = 'atom types'
    BOND_TYPES = 'bond types'
    ANGLE_TYPES = 'angle types'
    DIHE_TYPES = 'dihedral types'
    IMPROP_TYPES = 'improper types'
    TYPE_DSP = [ATOM_TYPES, BOND_TYPES, ANGLE_TYPES, DIHE_TYPES, IMPROP_TYPES]

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    BUFFER = [4., 4., 4.]

    MASSES = 'Masses'
    ATOM_ID = 'atom_id'
    TYPE_ID = oplsua.TYPE_ID

    ATOMS = 'atoms'
    BONDS = 'bonds'
    ANGLES = 'angles'
    DIHEDRALS = 'dihedrals'
    IMPROPERS = 'impropers'
    STRUCT_DSP = [ATOMS, BONDS, ANGLES, DIHEDRALS, IMPROPERS]

    MASSES = 'Masses'
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
    MIN_DIST = 1.4
    SCALE = 0.45


class Struct(structure.Struct, Base):

    MolClass = Mol

    def __init__(self, struct=None, ff=None, options=None, **kwargs):
        """
        :param struct Struct: struct object with moelcules and conformers.
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(struct=struct, **kwargs)
        Base.__init__(self, options=options, **kwargs)
        self.ff = ff
        self.total_charge = 0.
        self.atm_types = None
        self.bnd_types = None
        self.ang_types = None
        self.dihe_types = None
        self.impr_types = None
        self.hdl = None
        self.warnings = []
        self.excluded = collections.defaultdict(set)
        self.radii = None

    def addMol(self, mol):
        mol = super().addMol(mol)
        self.setTypeMap(mol)
        return mol

    def writeData(self, nofile=False):
        """
        Write out LAMMPS data file.

        :param nofile bool: return the string instead of writing to a file if True
        """

        with io.StringIO() if nofile else open(self.datafile, 'w') as self.hdl:
            self.reorderType()
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writePairCoeffs()
            self.writeBondCoeffs()
            self.writeAngleCoeffs()
            self.writeDiheCoeffs()
            self.writeImpropCoeffs()
            self.writeAtoms()
            self.writeBonds()
            self.writeAngles()
            self.writeDihedrals()
            self.writeImpropers()
            return self.getContents() if nofile else None

    def setTypeMap(self, mol):
        """
        Set the type map for atoms, bonds, angles, dihedrals, and impropers.
        """
        if self.atm_types is None:
            self.atm_types = np.zeros(max(self.ff.atoms) + 1, dtype=int)
        atypes = set(x.GetIntProp(self.TYPE_ID) for x in mol.GetAtoms())
        atypes = sorted(atypes.difference(np.nonzero(self.atm_types)[0]))
        start = self.atm_types.max() + 1
        self.atm_types[atypes] = list(range(start, start + len(atypes)))
        if self.bnd_types is None:
            self.bnd_types = np.zeros(max(self.ff.bonds) + 1, dtype=int)
        btypes = set(x[0] for x in mol.bonds)
        btypes = sorted(btypes.difference(np.nonzero(self.bnd_types)[0]))
        start = self.bnd_types.max() + 1
        self.bnd_types[btypes] = list(np.arange(start, start + len(btypes)))
        if self.ang_types is None:
            self.ang_types = np.zeros(max(self.ff.angles) + 1, dtype=int)
        antypes = set(y[0] for x in mol.GetConformers() for y in x.angles)
        antypes = sorted(antypes.difference(np.nonzero(self.ang_types)[0]))
        start = self.ang_types.max() + 1
        self.ang_types[antypes] = list(np.arange(start, start + len(antypes)))
        if self.dihe_types is None:
            self.dihe_types = np.zeros(max(self.ff.dihedrals) + 1, dtype=int)
        dtps = set(y[0] for x in mol.GetConformers() for y in x.dihedrals)
        dtps = sorted(dtps.difference(np.nonzero(self.dihe_types)[0]))
        start = self.dihe_types.max() + 1
        self.dihe_types[dtps] = list(np.arange(start, start + len(dtps)))
        if self.impr_types is None:
            self.impr_types = np.zeros(max(self.ff.impropers) + 1, dtype=int)
        itps = set(y[0] for x in mol.GetConformers() for y in x.impropers)
        itps = sorted(itps.difference(np.nonzero(self.impr_types)[0]))
        start = self.impr_types.max() + 1
        self.impr_types[itps] = list(np.arange(start, start + len(itps)))

    def reorderType(self):
        indexes = np.nonzero(self.atm_types)[0]
        self.atm_types[indexes] = np.arange(1, len(indexes) + 1)
        indexes = np.nonzero(self.bnd_types)[0]
        self.bnd_types[indexes] = np.arange(1, len(indexes) + 1)
        indexes = np.nonzero(self.ang_types)[0]
        self.ang_types[indexes] = np.arange(1, len(indexes) + 1)
        indexes = np.nonzero(self.dihe_types)[0]
        self.dihe_types[indexes] = np.arange(1, len(indexes) + 1)
        indexes = np.nonzero(self.impr_types)[0]
        self.impr_types[indexes] = np.arange(1, len(indexes) + 1)

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom,
        bond, angle etc.
        """
        lmp_dsp = self.LAMMPS_DESCRIPTION % self.atom_style
        self.hdl.write(f"{lmp_dsp}\n\n")
        self.hdl.write(f"{self.atom_total} {self.ATOMS}\n")
        self.hdl.write(f"{self.bond_total} {self.BONDS}\n")
        self.hdl.write(f"{self.angle_total} {self.ANGLES}\n")
        self.hdl.write(f"{self.dihedral_total} {self.DIHEDRALS}\n")
        self.hdl.write(f"{self.improper_total} {self.IMPROPERS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        self.hdl.write(f"{self.atm_types.max()} {self.ATOM_TYPES}\n")
        self.hdl.write(f"{self.bnd_types.max()} {self.BOND_TYPES}\n")
        self.hdl.write(f"{self.ang_types.max()} {self.ANGLE_TYPES}\n")
        self.hdl.write(f"{self.dihe_types.max()} {self.DIHE_TYPES}\n")
        self.hdl.write(f"{self.impr_types.max()} {self.IMPROP_TYPES}\n")
        self.hdl.write("\n")

    def writeBox(self, min_box=None, buffer=None):
        """
        Write box information.

        :param min_box list: minimum box size
        :param buffer list: buffer in three dimensions
        """
        xyzs = self.getPositions()
        ctr = xyzs.mean(axis=0)
        box_hf = self.getHalfBox(xyzs, min_box=min_box, buffer=buffer)
        box = [[x - y, x + y, z] for x, y, z in zip(ctr, box_hf, self.LO_HI)]
        if self.box is not None:
            boxes = zip(box, np.array(self.box).reshape(-1, 2))
            box = [[*x, symbols.POUND, *y] for x, y in boxes]
        for line in box:
            line = [f'{x:.2f}' if isinstance(x, float) else x for x in line]
            self.hdl.write(f"{' '.join(line)}\n")
        self.hdl.write("\n")
        if self.density is None:
            return
        # Calculate density as the revised box may alter the box size.
        vol = math.prod([x * 2 * nconstant.ANG_TO_CM for x in box_hf])
        density = self.molecular_weight / vol / scipy.constants.Avogadro
        if np.isclose(self.density, density):
            return
        msg = f'The density of the final data file is {density:.4g} kg/cm^3'
        self.warnings.append(msg)

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
            min_box = [1, 1, 1]
        if buffer is None:
            buffer = self.BUFFER  # yapf: disable
        box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
        if self.box is not None:
            box = [(x - y) for x, y in zip(self.box[1::2], self.box[::2])]
        box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]
        cut_off = min([self.options.lj_cut, self.options.coul_cut])
        if min(box_hf) < cut_off:
            # One particle interacts with another particle within cutoff twice:
            # within the box and across the PBC
            msg = f"The half box size ({min(box_hf):.2f} {symbols.ANGSTROM}) " \
                  f"is smaller than the {cut_off} {symbols.ANGSTROM} cutoff."
            self.warnings.append(msg)

        if self.conformer_total != 1:
            return box_hf
        # All-trans single molecule with internal tension runs into clashes
        # across PBCs and thus larger box is used.
        return [x * 1.2 for x in box_hf]

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.hdl.write(f"{self.MASSES}\n\n")
        indexes = np.nonzero(self.atm_types)[0]
        for oid, id in zip(indexes, self.atm_types[indexes]):
            atom = self.ff.atoms[oid]
            dscrptn = f"{atom.description} {atom.symbol} {oid}"
            self.hdl.write(f"{id} {atom.mass} # {dscrptn}\n")
        self.hdl.write(f"\n")

    def writePairCoeffs(self):
        """
        Write pair coefficients.
        """
        self.hdl.write(f"{self.PAIR_COEFFS}\n\n")
        indexes = np.nonzero(self.atm_types)[0]
        for oid, id in zip(indexes, self.atm_types[indexes]):
            vdw = self.ff.vdws[oid]
            self.hdl.write(f"{id} {vdw.ene:.4f} {vdw.dist:.4f}\n")
        self.hdl.write("\n")

    def writeBondCoeffs(self):
        """
        Write bond coefficients.
        """

        if not self.bnd_types.any():
            return

        self.hdl.write(f"{self.BOND_COEFFS}\n\n")
        indexes = np.nonzero(self.bnd_types)[0]
        for oid, id in zip(indexes, self.bnd_types[indexes]):
            bond = self.ff.bonds[oid]
            self.hdl.write(f"{id} {bond.ene} {bond.dist}\n")
        self.hdl.write("\n")

    def writeAngleCoeffs(self):
        """
        Write angle coefficients.
        """
        if not self.ang_types.any():
            return

        self.hdl.write(f"{self.ANGLE_COEFFS}\n\n")
        indexes = np.nonzero(self.ang_types)[0]
        for oid, id in zip(indexes, self.ang_types[indexes]):
            angle = self.ff.angles[oid]
            self.hdl.write(f"{id} {angle.ene} {angle.angle}\n")
        self.hdl.write("\n")

    def writeDiheCoeffs(self):
        """
        Write dihedral coefficients.
        """
        if not self.dihe_types.any():
            return

        self.hdl.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        indexes = np.nonzero(self.dihe_types)[0]
        for oid, id in zip(indexes, self.dihe_types[indexes]):
            params = [0., 0., 0., 0.]
            # LAMMPS: K1, K2, K3, K4 in 0.5*K1[1+cos(x)] + 0.5*K2[1-cos(2x)]...
            # OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            for ene_ang_n in self.ff.dihedrals[oid].constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if not params[ene_ang_n.n_parm]:
                    continue
                if (ene_ang_n.angle == 180.) ^ (not ene_ang_n.n_parm % 2):
                    params[ene_ang_n.n_parm] *= -1
            self.hdl.write(f"{id}  {' '.join(map(str, params))}\n")
        self.hdl.write("\n")

    def writeImpropCoeffs(self):
        """
        Write improper coefficients.
        """
        if not self.impr_types.any():
            return

        self.hdl.write(f"{self.IMPROPER_COEFFS}\n\n")
        indexes = np.nonzero(self.impr_types)[0]
        for oid, id in zip(indexes, self.impr_types[indexes]):
            impr = self.ff.impropers[oid]
            # LAMMPS: K in K[1+d*cos(nx)] vs OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            sign = 1 if impr.angle == 0. else -1
            self.hdl.write(f"{id} {impr.ene} {sign} {impr.n_parm}\n")
        self.hdl.write("\n")

    def writeAtoms(self, float_format='%.3f'):
        """
        Write atom coefficients.

        :param float_format str: the format of float (e.g. charges and
            coordinates).
        """

        self.hdl.write(f"{self.ATOMS.capitalize()}\n\n")
        self.atoms.to_csv(self.hdl,
                          sep=' ',
                          header=False,
                          float_format=float_format,
                          mode='a')
        self.hdl.write(f"\n")
        if not round(self.atoms[CHARGE].sum(), 4):
            return
        msg = f'The system has a net charge of {self.atoms[CHARGE].sum():.4f}'
        self.warnings.append(msg)

    def writeBonds(self, fmt='%i %i %i %i'):
        """
        Write bond coefficients.

        :param fmt str: the format of bond line in LAMMPS data file.
        """
        if not self.bond_total:
            return
        self.hdl.write(f"{self.BONDS.capitalize()}\n\n")
        bonds = [x.bonds for x in self.conformer if x.bonds.any()]
        bonds = np.concatenate(bonds, axis=0)
        bonds[:, 0] = self.bnd_types[bonds[:, 0]]
        ids = np.arange(bonds.shape[0]).reshape(-1, 1) + 1
        np.savetxt(self.hdl, np.concatenate([ids, bonds], axis=1), fmt=fmt)
        self.hdl.write(f"\n")

    def writeAngles(self, fmt='%i %i %i %i %i'):
        """
        Write angle coefficients.
        """
        if not self.angle_total:
            return
        self.hdl.write(f"{self.ANGLES.capitalize()}\n\n")
        # Some angles may be filtered out by improper
        angles = [x.angles for x in self.conformer if x.angles.any()]
        angles = np.concatenate(angles, axis=0)
        angles[:, 0] = self.ang_types[angles[:, 0]]
        ids = np.arange(angles.shape[0]).reshape(-1, 1) + 1
        np.savetxt(self.hdl, np.concatenate([ids, angles], axis=1), fmt=fmt)
        self.hdl.write(f"\n")

    def writeDihedrals(self, fmt='%i %i %i %i %i %i'):
        """
        Write dihedral coefficients.
        """
        if not self.dihedral_total:
            return

        self.hdl.write(f"{self.DIHEDRALS.capitalize()}\n\n")
        dihes = [x.dihedrals for x in self.conformer if x.dihedrals.any()]
        dihes = np.concatenate(dihes, axis=0)
        dihes[:, 0] = self.dihe_types[dihes[:, 0]]
        ids = np.arange(dihes.shape[0]).reshape(-1, 1) + 1
        np.savetxt(self.hdl, np.concatenate([ids, dihes], axis=1), fmt=fmt)
        self.hdl.write(f"\n")

    def writeImpropers(self, fmt='%i %i %i %i %i %i'):
        """
        Write improper coefficients.
        """
        if not self.improper_total:
            return

        self.hdl.write(f"{self.IMPROPERS.capitalize()}\n\n")
        imprps = [x.impropers for x in self.conformer if x.impropers.any()]
        imprps = np.concatenate(imprps, axis=0)
        imprps[:, 0] = self.impr_types[imprps[:, 0]]
        ids = np.arange(imprps.shape[0]).reshape(-1, 1) + 1
        np.savetxt(self.hdl, np.concatenate([ids, imprps], axis=1), fmt=fmt)
        self.hdl.write(f"\n")

    def getContents(self):
        """
        Return datafile contents in base64 encoding.

        :return `bytes`: the contents of the data file in base64 encoding.
        """
        self.hdl.seek(0)
        contents = base64.b64encode(self.hdl.read().encode("utf-8"))
        return b','.join([b'lammps_datafile', contents])

    def writeRun(self, *arg, **kwarg):
        """
        Write command to further equilibrate the system with molecules
        information considered.
        """
        btypes = [self.bnd_types[y] for x in self.molecules for y in x.fbonds]
        atypes = [self.ang_types[y] for x in self.molecules for y in x.fangles]
        testing = self.conformer_total == 1 and self.atom_total < 100
        struct_info = types.SimpleNamespace(btypes=btypes,
                                            atypes=atypes,
                                            testing=testing)
        super().writeRun(*arg, struct_info=struct_info, **kwarg)

    def setClashExclusion(self, mol, include14=True):
        """
        Bonded atoms and atoms in angles are in the exclusion. If include14=True,
        the dihedral angles are in the exclusion as well.

        :param include14 bool: If True, 1-4 interaction in a dihedral angle count
            as exclusion.
        """
        pairs = set()
        for conf in mol.GetConformers():
            bonds = [tuple(sorted(x[1:])) for x in conf.bonds]
            pairs = pairs.union(bonds)
            angles = [tuple(sorted(x[1::2])) for x in conf.angles]
            pairs = pairs.union(angles)
            imprps = [itertools.combinations(x[1:], 2) for x in conf.impropers]
            pairs = pairs.union([tuple(sorted(y)) for x in imprps for y in x])
            if include14:
                dihes = [tuple(sorted(x[1::3])) for x in conf.dihedrals]
                pairs = pairs.union(dihes)
        for id1, id2 in pairs:
            self.excluded[id1].add(id2)
            self.excluded[id2].add(id1)

    def setVdwRadius(self, mix=lammpsin.In.GEOMETRIC, scale=Base.SCALE):
        """
        Set the vdw radius based on the mixing rule and vdw radii.

        :param mix str: the mixing rules, including GEOMETRIC, ARITHMETIC, and
            SIXTHPOWER
        :param scale float: scale the vdw radius by this factor

        NOTE: the scaled radii here are more like diameters (or distance)
            between two sites.
        """
        if mix == lammpsin.In.GEOMETRIC:
            # Data.GEOMETRIC is optimized for speed and is supported
            atom_types = np.nonzero(self.atm_types)[0]
            radii = [0] + [self.ff.vdws[x].dist for x in atom_types]
            radii = np.full((len(radii), len(radii)), radii, dtype='float16')
            radii[:, 0] = radii[0, :]
            radii *= radii.transpose()
            radii = np.sqrt(radii)
            radii *= pow(2, 1 / 6) * scale
            radii[radii < self.MIN_DIST] = self.MIN_DIST
            id_map = {x: y for x, y in self.atoms[TYPE_ID].items()}
            self.radii = Radius(radii, id_map=id_map)
            return

    @property
    def bond_total(self):
        """
        Total number of bonds in the structure.

        :return int: Total number of bonds across all molecules and conformers.
        """
        return sum(x.bond_total for x in self.molecules)

    @property
    def angle_total(self):
        """
        Total number of angels in the structure.

        :return int: Total number of angels across all molecules and conformers.
        """
        return sum(x.angle_total for x in self.molecules)

    @property
    def dihedral_total(self):
        """
        Total number of dihedral angels in the structure.

        :return int: Total number of dihedral angels across all molecules and
            conformers.
        """
        return sum(x.dihedral_total for x in self.molecules)

    @property
    def improper_total(self):
        """
        Total number of improper angels in the structure.

        :return int: Total number of improper angels across all molecules and
            conformers.
        """
        return sum(x.improper_total for x in self.molecules)

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        return np.isclose(self.atoms[CHARGE], 0, 0.001).any()

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return sum([x.mw * x.GetNumConformers() for x in self.molecules])

    mw = molecular_weight

    @property
    def atoms(self):
        data = pd.concat(x.atoms for x in self.conformer)
        if self.atm_types is not None:
            data[TYPE_ID] = self.atm_types[data[TYPE_ID]]
        return data


class DataFileReader(Base):
    """
    LAMMPS Data file reader
    """

    def __init__(self, data_file=None, contents=None):
        """
        :param data_file str: data file with path
        :param contents `bytes`: parse the contents if data_file not provided.
        """
        self.data_file = data_file
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
            for y in self.LO_HI if y in self.lines[x]
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

    def setClashParams(self, include14=False, scale=Base.SCALE):
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

    def setVdwRadius(self, mix=lammpsin.In.GEOMETRIC, scale=Base.SCALE):
        """
        Set the vdw radius based on the mixing rule and vdw radii.

        :param mix str: the mixing rules, including GEOMETRIC, ARITHMETIC, and
            SIXTHPOWER
        :param scale float: scale the vdw radius by this factor

        NOTE: the scaled radii here are more like diameters (or distance)
            between two sites.
        """
        if mix == lammpsin.In.GEOMETRIC:
            # lammpsin.In.GEOMETRIC is optimized for speed and is supported
            atom_types = sorted(set([x.type_id for x in self.atoms.values()]))
            radii = [0] + [self.vdws[x].dist for x in atom_types]
            radii = np.full((len(radii), len(radii)), radii, dtype='float16')
            radii[:, 0] = radii[0, :]
            radii *= radii.transpose()
            radii = np.sqrt(radii)
            radii *= pow(2, 1 / 6) * scale
            radii[radii < self.MIN_DIST] = self.MIN_DIST
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
                if dist < self.MIN_DIST:
                    dist = self.MIN_DIST
                radii[id1][id2] = round(dist, 4)

        self.radii = collections.defaultdict(dict)
        for atom1 in self.atoms.values():
            for atom2 in self.atoms.values():
                self.radii[atom1.id][atom2.id] = radii[atom1.type_id][
                    atom2.type_id]
        self.radii = dict(self.radii)


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
