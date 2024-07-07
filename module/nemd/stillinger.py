import sh
import types
import numpy as np
from nemd import xtal
from nemd import symbols
from nemd import lammpsdata


class Struct(xtal.Struct):

    XYZ = 'XYZ'
    FORCE = 'force'
    CUSTOM_EXT = f'.{lammpsdata.Struct.DUMP}'

    def __init__(self, *args, tasks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = tasks
        self.units = self.METAL
        self.atom_style = self.ATOMIC
        self.elements = None
        if self.tasks is None:
            self.tasks = [self.XYZ]

    def initTypeMap(self):
        pass

    def writeLammpsIn(self):
        """
        Write out LAMMPS in script.
        """
        with open(self.lammps_in, 'w') as self.fh:
            self.setElements()
            self.writeSetup()
            self.readData()
            self.writePairStyle()
            self.writeDump()
            self.writeEnergy()

    def writeSetup(self):
        """
        Write in script description section.
        """
        self.fh.write(f"{self.UNITS} {self.units}\n")
        self.fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
        self.fh.write("boundary p p p\n")

    def writePairStyle(self):
        self.fh.write("pair_style sw\n")
        self.fh.write(f"pair_coeff * * {self.ff} Si\n")

    def writeDump(self):
        dump = f"{self.DUMP} 1 all custom 1 {self.lammps_dump} id "
        if self.XYZ in self.tasks:
            dump += "xu yu zu "
        if self.FORCE in self.tasks:
            dump += "fx fy fz "
        self.fh.write(f"{dump}\n")
        self.fh.write(f"dump_modify 1 format float '%20.15f'\n")

    def writeEnergy(self):
        self.fh.write("run 0\n")

    def writeData(self, *args, **kwargs):
        with open(self.datafile, 'w') as self.data_fh:
            self.setElements()
            self.writeCounting()
            self.writeBox()
            self.writeMasses()
            self.writeAtoms()

    def setElements(self):
        elements = [x.GetAtomicNum() for x in self.atom]
        self.elements = list(set(elements))

    def writeCounting(self):
        """
        Write the topology and type counting information.
        """
        self.data_fh.write(f"{self.DESCR.format(style=self.atom_style)}\n\n")
        atom_total = sum([x.GetNumAtoms() for x in self.molecules])
        self.data_fh.write(f"{atom_total} atoms\n\n")
        self.data_fh.write(f"{len(self.elements)} atom types\n\n")

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
            self.data_fh.write(
                f"{0:.4f} {box[dim]:.4f} {lammpsdata.Box.LO_HI[dim]}\n")
        # FIXME https://docs.lammps.org/Howto_triclinic.html
        self.data_fh.write("0.0000 0.0000 0.0000 xy xz yz\n")
        self.data_fh.write("\n")

    def writeMasses(self):
        """
        Write out mass information.
        """

        self.data_fh.write(f"{lammpsdata.Mass.NAME}\n\n")
        masses = list(set([x.GetMass() for x in self.atom]))
        for id, mass in enumerate(masses, 1):
            self.data_fh.write(f"{id} {mass}\n")
        self.data_fh.write(f"\n")

    def writeAtoms(self):
        """
        Write atom coefficients.
        """

        self.data_fh.write(f"{lammpsdata.Atom.NAME}\n\n")
        for mol in self.molecules:
            data = np.zeros((mol.GetNumAtoms(), 5))
            conformer = mol.GetConformer()
            aids = [x.GetIdx() for x in mol.GetAtoms()]
            data[:, 0] = conformer.id_map[aids]
            atomic_num = [x.GetAtomicNum() for x in mol.GetAtoms()]
            data[:, 1] = self.atm_types[atomic_num]
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
