import sh
import types
import numpy as np
from nemd import xtal
from nemd import symbols
from nemd import lammpsdata


class Data(xtal.Struct, lammpsdata.Base):

    XYZ = 'XYZ'
    FORCE = 'force'
    CUSTOM_EXT = f'.{lammpsdata.Base.DUMP}'

    def __init__(self, struct, *args, ff=None, tasks=None, **kwargs):
        xtal.Struct.__init__(self, struct, ff=ff)
        lammpsdata.Base.__init__(self, *args, **kwargs)
        self.tasks = tasks
        self.units = self.METAL
        self.atom_style = self.ATOMIC
        if self.tasks is None:
            self.tasks = [self.XYZ]

    def writeLammpsIn(self):
        """
        Write out LAMMPS in script.
        """
        with open(self.lammps_in, 'w') as self.in_fh:
            self.setElements()
            self.writeDescriptions()
            self.readData()
            self.writePairStyle()
            self.writeDump()
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

    def writeDump(self):
        dump = f"{self.DUMP} 1 all custom 1 {self.lammps_dump} id "
        if self.XYZ in self.tasks:
            dump += "xu yu zu "
        if self.FORCE in self.tasks:
            dump += "fx fy fz "
        self.in_fh.write(f"{dump}\n")
        self.in_fh.write(f"dump_modify 1 format float '%20.15f'\n")

    def writeEnergy(self):
        self.in_fh.write("run 0\n")

    def writeData(self, *args, **kwargs):
        with open(self.datafile, 'w') as self.data_fh:
            self.setElements()
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writeAtoms()

    def setElements(self):
        elements = [x.GetAtomicNum() for x in self.atoms]
        self.elements = list(set(elements))

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom, bond,
        angle etc.
        """
        lmp_dsp = self.LAMMPS_DESCRIPTION % self.atom_style
        self.data_fh.write(f"{lmp_dsp}\n\n")
        self.data_fh.write(f"{self.atom_total} {self.ATOMS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        self.data_fh.write(f"{len(self.elements)} {self.ATOM_TYPES}\n\n")

    def writeBox(self):
        """
        Write box information.
        """

        boxes = [x.getBox() for x in self.molecules]
        box = boxes[0]
        repeated = np.repeat(box.reshape(1, -1), len(boxes), axis=0)
        if not (repeated == boxes).all():
            raise ValueError("Unit cells have different PBCs.")
        for dim in range(3):
            self.data_fh.write(f"{0:.4f} {box[dim]:.4f} {self.LO_HI[dim]}\n")
        # FIXME https://docs.lammps.org/Howto_triclinic.html
        self.data_fh.write("0.0000 0.0000 0.0000 xy xz yz\n")
        self.data_fh.write("\n")

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.data_fh.write(f"{self.MASSES}\n\n")
        masses = list(set([x.GetMass() for x in self.atoms]))
        for id, mass in enumerate(masses, 1):
            self.data_fh.write(f"{id} {mass}\n")
        self.data_fh.write(f"\n")

    def writeAtoms(self):
        """
        Write atom coefficients.
        """

        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        for mol in self.molecules:
            data = np.zeros((mol.GetNumAtoms(), 5))
            conformer = mol.GetConformer()
            aids = [x.GetIdx() for x in mol.GetAtoms()]
            data[:, 0] = conformer.id_map[aids]
            data[:, 1] = conformer.gid
            data[:, 2:] = conformer.GetPositions()
            np.savetxt(self.data_fh, data, fmt='%i %i %.3f %.3f %.3f')
        self.data_fh.write(f"\n")


class DataFileReader(lammpsdata.DataFileReader):

    def setAtoms(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        sidx = self.mk_idxes[self.ATOMS_CAP] + 2
        for lid in range(sidx, sidx + self.struct_dsp[self.ATOMS]):
            id, type_id, x, y, z = self.lines[lid].split()[:5]
            self.atoms[int(id)] = types.SimpleNamespace(
                id=int(id),
                type_id=int(type_id),
                xyz=(float(x), float(y), float(z)),
                ele=self.masses[int(type_id)].ele)


def get_df_reader(data_file):
    line = sh.grep(lammpsdata.DataBase.LAMMPS_DESCRIPTION[:-2], data_file)
    if line.split(symbols.POUND)[1].strip() == lammpsdata.DataBase.ATOMIC:
        return DataFileReader(data_file)
    return lammpsdata.DataFileReader(data_file)
