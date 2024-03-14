# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Builder crystals.
"""
import crystals
import subprocess
from rdkit import Chem
from nemd import jobutils
from nemd import constants
from nemd import rdkitutils
from nemd import alamodeutils


class CrystalBuilder(object):

    EXECUTABLE = {
        alamodeutils.AlaWriter.SUGGEST: jobutils.ALM,
        alamodeutils.AlaWriter.PHONONS: jobutils.ANPHON,
        alamodeutils.AlaWriter.OPTIMIZE: jobutils.ALM
    }

    def __init__(self,
                 name,
                 jobname=None,
                 dim=constants.ONE_ONE_ONE,
                 scale_factor=constants.ONE_ONE_ONE):
        self.name = name
        self.jobname = jobname
        self.dim = dim
        self.scale_factor = scale_factor
        self.scell = None

    def run(self):
        """
        Main method to run.
        """
        self.setSuperCell()

    def setSuperCell(self):
        """
        Build supercell based on stretched or compressed supercell.
        """
        self.cell = crystals.Crystal.from_database(self.name)
        vect = [
            x * y for x, y in zip(self.cell.lattice_vectors, self.scale_factor)
        ]
        ucell_xtal = crystals.Crystal(self.cell, vect)
        self.scell = ucell_xtal.supercell(*self.dim)

    def getMol(self):
        """
        Return the crystal as a molecule.

        :return 'rdkit.Chem.rdchem.Mol': the molecule in the supercell.
        """
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

    def writeDispPattern(self):
        """
        Write the displacement pattern into a file.

        :return str: the filename containing displacement pattern.
        """
        filename = self.runAlm(mode=alamodeutils.AlaWriter.SUGGEST)
        ala_log_reader = alamodeutils.AlaLogReader(filename)
        return ala_log_reader.getDispPatternFile()

    def writeForceConstant(self):
        """
        Write the force constant into a file.

        :return str: the filename containing force constant.
        """
        filename = self.runAlm(mode=alamodeutils.AlaWriter.OPTIMIZE)
        ala_log_reader = alamodeutils.AlaLogReader(filename)
        return ala_log_reader.getAfcsXml()

    def writePhbands(self):
        """
        Write the phonon bands into a file.

        :return str: the filename containing phonon bands.
        """
        filename = self.runAlm(mode=alamodeutils.AlaWriter.PHONONS)
        ala_log_reader = alamodeutils.AlaLogReader(filename)
        return ala_log_reader.getPhBands()

    def runAlm(self, mode=alamodeutils.AlaWriter.SUGGEST):
        """
        Write input, Run alamode executable and return the logging for the
        submodule executable

        :param mode str: the alamode mode.
        :return str: filename containing the logging information.
        """
        cell = self.cell.primitive(
        ) if mode == alamodeutils.AlaWriter.PHONONS else self.scell
        ala_writer = alamodeutils.AlaWriter(cell,
                                            jobname=self.jobname,
                                            mode=mode)
        ala_writer.run()

        executable = self.EXECUTABLE[mode]
        cmd = f"{executable} {ala_writer.filename}"
        info = subprocess.run(cmd, capture_output=True, shell=True)

        ala_logfile = f'{ala_writer.jobname}_{mode.lower()}.log'
        with open(ala_logfile, 'wb') as fh:
            fh.write(info.stdout)

        if bool(info.stderr):
            raise ValueError(info.stderr)

        return ala_logfile
