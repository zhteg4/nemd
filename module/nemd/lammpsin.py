import os
import math
import string
import numpy as np

from nemd import symbols
from nemd import fileutils
from nemd import constants
from nemd import environutils
from nemd import lammpsfix


class FixWriter:
    """
    This the wrapper for LAMMPS fix command writer. which usually includes an
    unfix after the run command.
    """
    SET_VAR = lammpsfix.SET_VAR
    NVE = lammpsfix.NVE
    NVT = lammpsfix.NVT
    NPT = lammpsfix.NPT
    FIX = lammpsfix.FIX
    FIX_NVE = lammpsfix.FIX_NVE
    BERENDSEN = lammpsfix.BERENDSEN
    FIX_TEMP_BERENDSEN = lammpsfix.FIX_TEMP_BERENDSEN
    FIX_PRESS_BERENDSEN = lammpsfix.FIX_PRESS_BERENDSEN
    RUN_STEP = lammpsfix.RUN_STEP
    UNFIX = lammpsfix.UNFIX
    RECORD_BDRY = lammpsfix.RECORD_BDRY
    DUMP_EVERY = lammpsfix.DUMP_EVERY
    DUMP_ID = lammpsfix.DUMP_ID
    DUMP_Q = lammpsfix.DUMP_Q
    VOL = lammpsfix.VOL
    AMP = lammpsfix.AMP
    IMMED_PRESS = lammpsfix.IMMED_PRESS
    SET_IMMED_PRESS = lammpsfix.SET_IMMED_PRESS
    PRESS = lammpsfix.PRESS
    SET_PRESS = lammpsfix.SET_PRESS
    IMMED_MODULUS = lammpsfix.IMMED_MODULUS
    SET_IMMED_MODULUS = lammpsfix.SET_IMMED_MODULUS
    MODULUS = lammpsfix.MODULUS
    SET_MODULUS = lammpsfix.SET_MODULUS
    FACTOR = lammpsfix.FACTOR
    SET_FACTOR = lammpsfix.SET_FACTOR
    SET_LABEL = lammpsfix.SET_LABEL
    FIX_DEFORM = lammpsfix.FIX_DEFORM
    WIGGLE_VOL = lammpsfix.WIGGLE_VOL
    RECORD_PRESS_VOL = lammpsfix.RECORD_PRESS_VOL
    CHANGE_BDRY = lammpsfix.CHANGE_BDRY
    SET_LOOP = lammpsfix.SET_LOOP
    MKDIR = lammpsfix.MKDIR
    CD = lammpsfix.CD
    JUMP = lammpsfix.JUMP
    IF_JUMP = lammpsfix.IF_JUMP
    PRINT = lammpsfix.PRINT
    NEXT = lammpsfix.NEXT
    DEL_VAR = lammpsfix.DEL_VAR
    PRESS_VAR = f'${{{PRESS}}}'
    MODULUS_VAR = f'${{{MODULUS}}}'

    def __init__(self, fh, options=None, testing=True):
        """
        :param fh '_io.TextIOWrapper': file handdle to write fix commands
        :param options 'argparse.Namespace': command line options
        :param testing bool: the structure object.
        """
        self.fh = fh
        self.testing = testing
        self.options = options
        self.cmd = []
        self.timestep = self.options.timestep
        self.relax_time = self.options.relax_time
        self.prod_time = self.options.prod_time
        self.stemp = self.options.stemp
        self.temp = self.options.temp
        self.tdamp = self.options.timestep * self.options.tdamp
        self.press = self.options.press
        self.pdamp = self.options.timestep * self.options.pdamp
        ps_timestep = self.timestep / constants.NANO_TO_FEMTO
        self.relax_step = round(self.relax_time / ps_timestep)
        self.prod_step = round(self.prod_time / ps_timestep)

    def run(self):
        """
        Main method to run the writer.
        """
        self.test()
        self.startLow()
        self.rampUp()
        self.relaxAndDefrom()
        self.production()
        self.write()

    def test(self, nstep=1E3):
        """
        Append command for testing and conformal search (more effect needed).

        :nstep int: run this steps for time integration.
        """
        if not self.testing:
            return
        self.nve(nstep=nstep)

    def nve(self, nstep=1E3):
        """
        Append command for constant energy and volume.

        :nstep int: run this steps for time integration.
        """
        # NVT on single molecule gives nan coords (guess due to translation)
        cmd = self.FIX_NVE + self.RUN_STEP % nstep + self.UNFIX
        self.cmd.append(cmd)

    def startLow(self):
        """
        Start simulation from low temperature and constant volume.
        """
        if self.testing:
            return
        self.nvt(nstep=self.relax_step / 1E3,
                 stemp=self.stemp,
                 temp=self.stemp)

    def nvt(self, nstep=1E4, stemp=300, temp=300, style=BERENDSEN, pre=''):
        """
        Append command for constant volume and temperature.

        :nstep int: run this steps for time integration
        :stemp float: starting temperature
        :temp float: target temperature
        :style str: the style for the command
        :pre str: additional pre-conditions
        """
        if style == self.BERENDSEN:
            cmd1 = self.FIX_TEMP_BERENDSEN.format(stemp=stemp,
                                                  temp=temp,
                                                  tdamp=self.tdamp)
            cmd2 = self.FIX_NVE
        cmd = pre + cmd1 + cmd2
        fix = [x for x in cmd.split(symbols.RETURN) if x.startswith(self.FIX)]
        self.cmd.append(cmd + self.RUN_STEP % nstep + self.UNFIX * len(fix))

    def rampUp(self, ensemble=None):
        """
        Ramp up temperature to the targe value.

        :ensemble str: the ensemble to ramp up temperature.

        NOTE: ensemble=None runs NVT at low temperature and ramp up with constant
        volume, calculate the averaged pressure at high temperature, and changes
        volume to reach the target pressure.
        """
        if self.testing:
            return
        if ensemble == self.NPT:
            self.npt(nstep=self.relax_step / 1E1,
                     stemp=self.stemp,
                     temp=self.temp,
                     press=self.press)
            return

        self.nvt(nstep=self.relax_step / 1E2, stemp=self.stemp, temp=self.temp)
        self.cycleToPress()
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.temp,
                 temp=self.temp,
                 spress=self.PRESS_VAR,
                 press=self.press,
                 modulus=self.MODULUS_VAR)

    def npt(self,
            nstep=20000,
            stemp=300,
            temp=300,
            spress=1.,
            press=1.,
            style=BERENDSEN,
            modulus=10,
            pre=''):
        """
        Append command for constant pressure and temperature.

        :nstep int: run this steps for time integration
        :stemp int: starting temperature
        :temp float: target temperature
        :spress float: starting pressure
        :press float: target pressure
        :style str: the style for the command
        :pre str: additional pre-conditions
        """
        if spress is None:
            spress = press
        if style == self.BERENDSEN:
            cmd1 = self.FIX_PRESS_BERENDSEN.format(spress=spress,
                                                   press=press,
                                                   pdamp=self.pdamp,
                                                   modulus=modulus)
            cmd2 = self.FIX_TEMP_BERENDSEN.format(stemp=stemp,
                                                  temp=temp,
                                                  tdamp=self.tdamp)
            cmd3 = self.FIX_NVE
        cmd = pre + cmd1 + cmd2 + cmd3
        fix = [x for x in cmd.split(symbols.RETURN) if x.startswith(self.FIX)]
        self.cmd.append(cmd + self.RUN_STEP % nstep + self.UNFIX * len(fix))

    def cycleToPress(self,
                     max_loop=100,
                     num=3,
                     record_num=100,
                     defm_id='defm_id',
                     defm_start='defm_start',
                     defm_break='defm_break'):
        """
        Deform the box by cycles to get close to the target pressure.
        One cycle consists of sinusoidal wave, print properties, deformation,
        and relaxation. The max total simulation time for the all cycles is the
        regular relaxation simulation time.

        :param max_loop int: the maximum number of big cycle loops.
        :param num int: the number of sinusoidal cycles.
        :param record_num int: each sinusoidal wave records this number of data.
        :param defm_id str: Deformation id loop from 0 to max_loop - 1
        :param defm_start str: Each deformation loop starts with this label
        :param defm_break str: Terminate the loop by go to the this label
        """
        # The number of steps for one sinusoidal cycle that yields 10 records
        nstep = int(self.relax_step / max_loop / (num + 1))
        nstep = max([int(nstep / record_num), 10]) * record_num
        cyc_nstep = nstep * (num + 1)
        # Each cycle dumps one trajectory frame
        self.cmd.append(self.DUMP_EVERY.format(id=self.DUMP_ID, arg=cyc_nstep))
        # Set variables used in the loop
        self.cmd.append(self.SET_VAR.format(var=self.VOL, expr=self.VOL))
        expr = f'0.01*v_{self.VOL}^(1/3)'
        self.cmd.append(self.SET_VAR.format(var=self.AMP, expr=expr))
        self.cmd.append(self.SET_IMMED_PRESS)
        self.cmd.append(self.SET_IMMED_MODULUS.format(record_num=record_num))
        self.cmd.append(self.SET_FACTOR.format(press=self.options.press))
        self.cmd.append(self.SET_LOOP.format(id=defm_id, end=max_loop - 1))
        self.cmd.append(self.SET_LABEL.format(label=defm_start))
        self.cmd.append(self.PRINT.format(var=defm_id))
        # Run in a subdirectory as some output files are of the same names
        dirname = f"defm_${{{defm_id}}}"
        self.cmd.append(self.MKDIR.format(dir=dirname))
        self.cmd.append(self.CD.format(dir=dirname))
        self.cmd.append("")
        pre = self.getCyclePre(nstep, record_num=record_num)
        self.nvt(nstep=nstep * num, stemp=self.temp, temp=self.temp, pre=pre)
        self.cmd.append(self.PRINT.format(var=self.IMMED_PRESS))
        self.cmd.append(self.PRINT.format(var=self.IMMED_MODULUS))
        self.cmd.append(self.PRINT.format(var=self.FACTOR))
        cond = f"${{{defm_id}}} == {max_loop - 1} || ${{{self.FACTOR}}} == 1"
        self.cmd.append(self.IF_JUMP.format(cond=cond, label=defm_break))
        self.cmd.append("")
        self.nvt(nstep=nstep / 2,
                 stemp=self.temp,
                 temp=self.temp,
                 pre=self.FIX_DEFORM)
        self.nvt(nstep=nstep / 2, stemp=self.temp, temp=self.temp)
        self.cmd.append(self.CD.format(dir=os.pardir))
        self.cmd.append(self.NEXT.format(id=defm_id))
        self.cmd.append(self.JUMP.format(label=defm_start))
        self.cmd.append("")
        self.cmd.append(self.SET_LABEL.format(label=defm_break))
        # Record press and modulus as immediate variable evaluation uses files
        self.cmd.append(self.SET_MODULUS)
        self.cmd.append(self.SET_PRESS)
        self.cmd.append(self.CD.format(dir=os.pardir))
        # Delete variables used in the loop
        self.cmd.append(self.DEL_VAR.format(var=self.VOL))
        self.cmd.append(self.DEL_VAR.format(var=self.AMP))
        self.cmd.append(self.DEL_VAR.format(var=self.IMMED_PRESS))
        self.cmd.append(self.DEL_VAR.format(var=self.IMMED_MODULUS))
        self.cmd.append(self.DEL_VAR.format(var=self.FACTOR))
        self.cmd.append(self.DEL_VAR.format(var=defm_id))
        # Restore dump defaults
        cmd = '\n' + self.DUMP_EVERY.format(id=self.DUMP_ID, arg=self.DUMP_Q)
        self.cmd.append(cmd)

    def getCyclePre(self, nstep, record_num=100):
        """
        Get the pre-stage str for the cycle simulation.

        :param nstep int: the simulation steps of the one cycles
        :param record_num int: each cycle records this number of data
        :return str: the prefix string of the cycle stage.
        """

        wiggle = self.WIGGLE_VOL.format(period=nstep * self.timestep)
        record_period = int(nstep / record_num)
        record_press = self.RECORD_PRESS_VOL.format(period=record_period)
        return record_press + wiggle

    def relaxAndDefrom(self):
        """
        Longer relaxation at constant temperature and deform to the mean size.
        """
        if self.testing:
            return
        if self.options.prod_ens == self.NPT:
            self.npt(nstep=self.relax_step,
                     stemp=self.temp,
                     temp=self.temp,
                     press=self.press,
                     modulus=self.MODULUS_VAR)
            return
        # NVE and NVT production runs use averaged cell
        pre = self.getBdryPre()
        self.npt(nstep=self.relax_step,
                 stemp=self.temp,
                 temp=self.temp,
                 press=self.press,
                 modulus=self.MODULUS_VAR,
                 pre=pre)
        self.cmd.append(self.CHANGE_BDRY)
        self.nvt(nstep=self.relax_step / 1E2, stemp=self.temp, temp=self.temp)

    def production(self):
        """
        Production run. NVE is good for all, specially transport properties, but
        requires for good energy conservation in time integration. NVT and NPT
        may help for disturbance non-sensitive property.
        """
        if self.testing:
            return
        if self.options.prod_ens == self.NVE:
            self.nve(nstep=self.prod_step)
        elif self.options.prod_ens == self.NVT:
            self.nvt(nstep=self.prod_step, stemp=self.temp, temp=self.temp)
        else:
            self.npt(nstep=self.prod_step,
                     stemp=self.temp,
                     temp=self.temp,
                     press=self.press,
                     modulus=self.MODULUS_VAR)

    def getBdryPre(self, start_pct=0.2):
        """
        Record the boundary during the time integration before unset.

        :param start_pct float: exclude the first this percentage from average
        return str: the prefix str to get the box boundary.
        """
        start = math.floor(start_pct * self.relax_step)
        every = self.relax_step - start
        return self.RECORD_BDRY.format(start=start, every=every)

    def write(self):
        """
        Write the command to the file.
        """
        for idx, cmd in enumerate(self.cmd, 1):
            num = round(cmd.count('%s') / 2)
            ids = [f'{idx}{string.ascii_lowercase[x]}' for x in range(num)]
            ids += [x for x in reversed(ids)]
            cmd = cmd % tuple(ids) if ids else cmd
            self.fh.write(cmd + '\n')


class In(fileutils.LammpsInput):
    """
    Class to write out LAMMPS in script.
    """

    IN_EXT = '.in'
    DATA_EXT = '.data'

    METAL = 'metal'
    ATOMIC = 'atomic'

    LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
    LJ_CUT = 'lj/cut'
    GEOMETRIC = 'geometric'
    ARITHMETIC = 'arithmetic'
    SIXTHPOWER = 'sixthpower'

    MIX = 'mix'
    PPPM = 'pppm'
    REAL = 'real'
    FULL = 'full'
    OPLS = 'opls'
    CVFF = 'cvff'
    FIRE = 'fire'
    MIN_STYLE = 'min_style'
    HARMONIC = 'harmonic'
    LJ_COUL = 'lj/coul'
    CUSTOM_EXT = '.custom.gz'
    DUMP = 'dump'
    DEFAULT_CUT = 11.
    DEFAULT_LJ_CUT = DEFAULT_CUT
    DEFAULT_COUL_CUT = DEFAULT_CUT
    DUMP_ID, DUMP_Q = FixWriter.DUMP_ID, FixWriter.DUMP_Q
    FIX_RIGID_SHAKE = 'fix rigid all shake 0.0001 10 10000 b {bond} a {angle}\n'

    def __init__(self, jobname='tmp', options=None):
        """
        :param jobname str: jobname based on which out filenames are defined
        :param options 'argparse.Namespace': command line options
        """
        self.jobname = jobname
        self.options = options
        self.lammps_in = self.jobname + self.IN_EXT
        self.datafile = self.jobname + self.DATA_EXT
        self.lammps_dump = self.jobname + self.CUSTOM_EXT
        self.units = self.REAL
        self.atom_style = self.FULL
        self.bond_style = self.HARMONIC
        self.angle_style = self.HARMONIC
        self.dihedral_style = self.OPLS
        self.improper_style = self.CVFF
        self.lj_cut = getattr(self.options, 'lj_cut', self.DEFAULT_LJ_CUT)
        self.coul_cut = getattr(self.options, 'coul_cut',
                                self.DEFAULT_COUL_CUT)
        lj_coul_cut = f"{self.lj_cut} {self.coul_cut}"
        self.pair_style = {
            self.LJ_CUT: f"{self.LJ_CUT} {self.lj_cut}",
            self.LJ_CUT_COUL_LONG: f"{self.LJ_CUT_COUL_LONG} {lj_coul_cut}"
        }
        self.in_fh = None
        self.is_debug = environutils.is_debug()

    def resetFilenames(self, jobname):
        """
        Reset the filenames based on the new jobname.

        "param jobname str: new jobname based on which out filenames are defined
        """
        self.lammps_in = jobname + self.IN_EXT
        self.datafile = jobname + self.DATA_EXT
        self.lammps_dump = jobname + self.CUSTOM_EXT

    def writeIn(self):
        """
        Write out LAMMPS in script.
        """
        with open(self.lammps_in, 'w') as self.in_fh:
            self.writeDescriptions()
            self.readData()
            self.writeTimestep()
            self.writeMinimize()
            self.writeFixShake()
            self.writeRun()

    def writeDescriptions(self):
        """
        Write in script description section.
        """
        self.in_fh.write(f"{self.UNITS} {self.units}\n")
        self.in_fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
        self.in_fh.write(f"{self.BOND_STYLE} {self.bond_style}\n")
        self.in_fh.write(f"{self.ANGLE_STYLE} {self.angle_style}\n")
        self.in_fh.write(f"{self.DIHEDRAL_STYLE} {self.dihedral_style}\n")
        self.in_fh.write(f"{self.IMPROPER_STYLE} {self.improper_style}\n")
        pair_style = self.LJ_CUT_COUL_LONG if self.hasCharge() else self.LJ_CUT
        self.in_fh.write(f"{self.PAIR_STYLE} {self.pair_style[pair_style]}\n")
        self.in_fh.write(f"{self.PAIR_MODIFY} {self.MIX} {self.GEOMETRIC}\n")
        self.in_fh.write(f"{self.SPECIAL_BONDS} {self.LJ_COUL} 0 0 0.5\n")
        if self.hasCharge():
            self.in_fh.write(f"{self.KSPACE_STYLE} {self.PPPM} 0.0001\n")

    def writeFixShake(self, bond=None, angle=None):
        """
        Write fix shake command to enforce constant bond length and angel values.

        :param bond float: bond types to be enforced.
        :param angle float: angle types to be enforced.
        """
        if not any([bond, angle]):
            return
        self.in_fh.write(self.FIX_RIGID_SHAKE.format(bond=bond, angle=angle))

    def hasCharge(self):
        """
        Whether any atom has non-zero charge. This method should be overwritten
        when force field and structure are available.

        :return bool: True if any atom has non-zero charge.
        """

        return True

    def readData(self):
        """
        Write data file related information.
        """
        self.in_fh.write(f"{self.READ_DATA} {self.datafile}\n\n")

    def writeMinimize(self, min_style=FIRE, dump=True):
        """
        Write commands related to minimization.

        :param min_style str: cg, fire, spin, etc.
        :param dump bool: Whether dump out trajectory.
        """
        if dump:
            self.in_fh.write(
                f"{self.DUMP} {self.DUMP_ID} all custom {self.DUMP_Q} "
                f"dump{self.CUSTOM_EXT} id xu yu zu\n")
            self.in_fh.write("dump_modify 1 sort id\n")
        self.in_fh.write(f"{self.MIN_STYLE} {min_style}\n")
        self.in_fh.write("minimize 1.0e-6 1.0e-8 1000000 10000000\n\n")

    def writeTimestep(self):
        """
        Write commands related to timestep.
        """
        self.in_fh.write(f'timestep {self.options.timestep}\n')
        self.in_fh.write('thermo_modify flush yes\n')
        self.in_fh.write('thermo 1000\n')

    def dumpImproper(self):
        """
        Compute and dump improper values with type.
        """
        self.in_fh.write(
            'compute 1 all property/local itype iatom1 iatom2 iatom3 iatom4\n')
        self.in_fh.write('compute 2 all improper/local chi\n')
        self.in_fh.write('dump 1i all local 1000 tmp.dump index c_1[1] c_2\n')

    def writeRun(self, testing=True):
        """
        Write command to further equilibrate the system.

        :param testing bool: fixes are for testing only.
        """
        seed = np.random.randint(0, high=constants.LARGE_NUM)
        self.in_fh.write(f"velocity all create {self.options.stemp} {seed}\n")
        fwriter = FixWriter(self.in_fh, options=self.options, testing=testing)
        fwriter.run()
