import numpy as np
from nemd import oplsua


class LammpsData(oplsua.LammpsData):

    METAL = 'metal'
    ATOMIC = 'atomic'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = self.METAL
        self.atom_style = self.ATOMIC

    def writeLammpsIn(self):
        """
        Write out LAMMPS in script.
        """
        with open(self.lammps_in, 'w') as self.in_fh:
            self.writeDescriptions()
            self.readData()
            self.writePairStyle()
            self.writeData()
            self.writeEnergy()

    def writeDescriptions(self):
        """
        Write in script description section.
        """
        self.in_fh.write(f"{self.UNITS} {self.units}\n")
        self.in_fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
        self.in_fh.write("boundary p p p\n")

    def writePairStyle(self):
        self.in_fh.write("pair_style sw\n")
        self.in_fh.write(f"pair_coeff * * {self.ff} Si\n")

    def writeEnergy(self):
        self.in_fh.write("run 0\n")

    def writeData(self, adjust_coords=False):

        with open(self.lammps_data, 'w') as self.data_fh:
            self.setAtoms()
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writeAtoms()

    def setAtoms(self):
        super().setAtoms()
        elements = [
            y.GetAtomicNum() for x in self.mols.values() for y in x.GetAtoms()
        ]
        self.elements = list(set(elements))

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom, bond,
        angle etc.
        """
        if self.mols is None:
            raise ValueError(f"Mols are not set.")

        self.data_fh.write(f"{self.LAMMPS_DESCRIPTION}\n\n")
        atom_nums = [len(x.GetAtoms()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(atom_nums)} {self.ATOMS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        self.data_fh.write(f"{len(self.elements)} {self.ATOM_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        """
        Write box information.

        :param min_box list: minimum box size
        :param buffer list: buffer in three dimensions
        """

        boxes = [x.getBox() for x in self.mols.values()]
        if all([x is not None for x in boxes]):
            box = boxes[0]
            repeated = np.repeat(box.reshape(1, -1), len(boxes), axis=0)
            if (repeated == boxes).all():
                for dim in range(3):
                    self.data_fh.write(
                        f"{0:.4f} {box[dim]:.4f} {self.LO_HI[dim]}\n")
                # FIXME https://docs.lammps.org/Howto_triclinic.html
                self.data_fh.write("0.0000 0.0000 0.0000 xy xz yz\n")
                self.data_fh.write("\n")
                return
        super().writeBox()

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.data_fh.write(f"{self.MASSES}\n\n")
        masses = list(
            set([
                y.GetMass() for x in self.mols.values() for y in x.GetAtoms()
            ]))
        for id, mass in enumerate(masses, 1):
            self.data_fh.write(f"{id} {mass}\n")
        self.data_fh.write(f"\n")

    def writeAtoms(self):
        """
        Write atom coefficients.

        :param comments bool: If True, additional descriptions including element
            sysmbol are written after each atom line
        """

        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        for mol_id, mol in self.mols.items():
            data = np.zeros((mol.GetNumAtoms(), 5))
            conformer = mol.GetConformer()
            data[:, 0] = [x.GetIntProp(self.ATOM_ID) for x in mol.GetAtoms()]
            data[:, 1] = mol_id
            data[:, 2:] = conformer.GetPositions()
            np.savetxt(self.data_fh, data, fmt='%i %i %.3f %.3f %.3f')
        self.data_fh.write(f"\n")
