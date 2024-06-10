# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Builder crystals.
"""
import os
import spglib
import pathlib
import crystals
import subprocess
import numpy as np
from rdkit import Chem
from nemd import jobutils
from nemd import constants
from nemd import structure
from nemd import alamodeutils


class Mol(structure.Mol):

    LATTICE_PARAMETERS = 'lattice_params'
    DIMENSIONS = 'dimensions'

    def __init__(self, mol, *args, **kwargs):
        lt_params = kwargs.pop(self.LATTICE_PARAMETERS, None)
        dimensions = kwargs.pop(self.DIMENSIONS, None)
        super().__init__(mol, *args, **kwargs)
        self.lattice_params = lt_params if lt_params else mol.lattice_params
        self.dimensions = dimensions if dimensions else mol.dimensions

    def getBox(self):
        if self.lattice_params is None:
            return
        param = self.lattice_params[:3]
        return np.array(param) * self.dimensions


class Struct(structure.Struct):

    MolClass = Mol


class CrystalBuilder(object):

    EXECUTABLE = {
        alamodeutils.AlaWriter.SUGGEST: jobutils.ALM,
        alamodeutils.AlaWriter.PHONONS: jobutils.ANPHON,
        alamodeutils.AlaWriter.OPTIMIZE: jobutils.ALM
    }

    LD_LIBRARY_PATH = {
        'LD_LIBRARY_PATH': pathlib.Path(spglib.__path__[0]).joinpath('lib')
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
        mol = structure.Mol(emol.GetMol())
        cfm = structure.Conformer(mol.GetNumAtoms())
        [cfm.SetAtomPosition(x, atoms[x].coords_cartesian) for x in idxs]
        mol.AddConformer(cfm)
        return Mol(mol,
                   lattice_params=self.scell.lattice_parameters,
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
        env = os.environ.copy()
        env.update(self.LD_LIBRARY_PATH)
        info = subprocess.run(cmd, capture_output=True, shell=True, env=env)

        ala_logfile = f'{ala_writer.jobname}_{mode.lower()}.log'
        with open(ala_logfile, 'wb') as fh:
            fh.write(info.stdout)

        if bool(info.stderr) and info.returncode != 0:
            raise ValueError(info.stderr)
        # The following stderr error message as info.stderr is more of a warning
        # '[xxx.local:45424] shmem: mmap: an error occurred while determining
        # whether or not xxx could be created.\n'
        return ala_logfile
