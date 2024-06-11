import math
import string
from scipy import constants

from nemd import symbols
from nemd import fileutils
from nemd import environutils

NVT = 'NVT'
NPT = 'NPT'
NVE = 'NVE'
ENSEMBLES = [NVE, NVT, NPT]


class FixWriter:
    """
    This the wrapper for LAMMPS fix command writer. which usually includes a
    unfix after the run command.
    """

    VOL = 'vol'
    PRESS = 'press'
    MODULUS = 'modulus'
    IMMED_MODULUS = 'immed_modulus'
    AVE_PRESS = 'ave_press'
    IMMED_PRESS = 'immed_press'
    FACTOR = 'factor'
    TEMP_BERENDSEN = 'temp/berendsen'
    PRESS_BERENDSEN = f'{PRESS}/berendsen'
    FIX = 'fix'
    SET_LABEL = "label %s"
    DEL_VARIABLE = "variable %s delete"
    DUMP_EVERY = "dump_modify {id} every {arg}"
    DUMP_ID, DUMP_Q = 1, 1000

    RUN_STEP = "run %i\n"
    UNFIX = "unfix %s\n"
    FIX_NVE = f"{FIX} %s all nve\n"
    FIX_NVT = f"{FIX} %s all nvt temp {{stemp}} {{temp}} {{tdamp}}\n"
    FIX_TEMP_BERENDSEN = f"{FIX} %s all {TEMP_BERENDSEN} {{stemp}} {{temp}} {{tdamp}}\n"
    FIX_PRESS_BERENDSEN = f"{FIX} %s all {PRESS_BERENDSEN} iso {{spress}} {{press}} {{pdamp}} {MODULUS} {{modulus}}\n"
    FIX_DEFORM = f"{FIX} %s all deform 100 x scale ${{factor}} y scale ${{factor}} z scale ${{factor}} remap v\n"
    PRESS_VOL_FILE = 'press_vol.data'
    SET_VOL = f"variable {VOL} equal {VOL}"
    RECORD_PRESS_VOL = f"{FIX} %s all ave/time 1 {{period}} {{period}} " \
                   f"c_thermo_{PRESS} v_{VOL} file {PRESS_VOL_FILE}\n"
    WIGGLE_DIM = "%s wiggle ${{amp}} {period}"
    AMP = 'amp'
    VARIABLE_AMP = f'variable {AMP} equal "0.01*{VOL}^(1/3)"\n'
    WIGGLE_VOL = f"{FIX} %s all deform 100 {{PARAM}}\n"

    SET_MODULUS = f"""
    variable {IMMED_MODULUS} python getModulus
    python getModulus input 2 {PRESS_VOL_FILE} %s return v_{IMMED_MODULUS} format sif here "from nemd.pyfunc import getModulus"
    """
    SET_MODULUS = SET_MODULUS.replace('\n    ', '\n').lstrip('\n')

    SET_PRESS = f"""
    variable {IMMED_PRESS} python getPress
    python getPress input 1 {PRESS_VOL_FILE} return v_{IMMED_PRESS} format sf here "from nemd.pyfunc import getPress"
    """
    SET_PRESS = SET_PRESS.replace('\n    ', '\n').lstrip('\n')

    SET_FACTOR = f"""
    variable {FACTOR} python getBdryFactor
    python getBdryFactor input 2 %f press_vol.data return v_{FACTOR} format fsf here "from nemd.pyfunc import getBdryFactor"
    """
    SET_FACTOR = SET_FACTOR.replace('\n    ', '\n').lstrip('\n')

    XYZL_FILE = 'xyzl.data'
    RECORD_BDRY = f"""
    variable xl equal "xhi - xlo"
    variable yl equal "yhi - ylo"
    variable zl equal "zhi - zlo"
    fix %s all ave/time 1 1000 1000 v_xl v_yl v_zl file {XYZL_FILE}
    """
    RECORD_BDRY = RECORD_BDRY.replace('\n    ', '\n').lstrip('\n')

    CHANGE_BOX = "change_box all x scale ${factor} y scale ${factor} z scale ${factor} remap\n"
    CHANGE_BDRY = f"""
    print "Final Boundary: xl = ${{xl}}, yl = ${{yl}}, zl = ${{zl}}"
    variable ave_xl python getXL
    python getXL input 1 {XYZL_FILE} return v_ave_xl format sf here "from nemd.pyfunc import getXL"
    variable ave_yl python getYL
    python getYL input 1 {XYZL_FILE} return v_ave_yl format sf here "from nemd.pyfunc import getYL"
    variable ave_zl python getZL
    python getZL input 1 {XYZL_FILE} return v_ave_zl format sf here "from nemd.pyfunc import getZL"
    print "Averaged  xl = ${{ave_xl}} yl = ${{ave_yl}} zl = ${{ave_zl}}"\n
    variable ave_xr equal "v_ave_xl / v_xl"
    variable ave_yr equal "v_ave_yl / v_yl"
    variable ave_zr equal "v_ave_zl / v_zl"
    change_box all x scale ${{ave_xr}} y scale ${{ave_yr}} z scale ${{ave_zr}} remap
    variable ave_xr delete
    variable ave_yr delete
    variable ave_zr delete
    variable ave_xl delete
    variable ave_yl delete
    variable ave_zl delete
    variable xl delete
    variable yl delete
    variable zl delete
    """
    CHANGE_BDRY = CHANGE_BDRY.replace('\n    ', '\n').lstrip('\n')

    def __init__(self, fh, options=None, mols=None, struct=None):
        """
        :param fh '_io.TextIOWrapper': file handdle to write fix commands
        :param options 'argparse.Namespace': command line options
        :param mols dict: id and rdkit.Chem.rdchem.Mol
        """
        self.fh = fh
        self.options = options
        self.mols = mols
        self.struct = struct
        self.cmd = []
        self.mols = {} if mols is None else mols
        self.mol_num = len(self.struct.mols)
        self.atom_num = self.struct.atom_total
        self.testing = self.mol_num == 1 and self.atom_num < 100
        self.timestep = self.options.timestep
        self.relax_time = self.options.relax_time
        self.prod_time = self.options.prod_time
        self.stemp = self.options.stemp
        self.temp = self.options.temp
        self.tdamp = self.options.timestep * self.options.tdamp
        self.press = self.options.press
        self.pdamp = self.options.timestep * self.options.pdamp
        nano_femto = constants.nano / constants.femto
        self.relax_step = round(self.relax_time / self.timestep * nano_femto)
        self.prod_step = round(self.prod_time / self.timestep * nano_femto)

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
        if ensemble == NPT:
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
                 spress='${ave_press}',
                 press=self.press,
                 modulus="${modulus}")

    def cycleToPress(self, max_loop=100, num=3, record_num=100):
        """
        Deform the box by cycles to get close to the target pressure.

        :param max_loop int: the maximum number of big cycle loops.
        :param num int: the number of sinusoidal cycles.
        :param record_num int: each sinusoidal wave records this number of data.
        """
        # Sinusoidal wave, print properties, cycle deformation, cycle relaxation
        # The max simulation time for the three stages is the regular relaxation
        nstep = int(self.relax_step / max_loop / (num + 1))
        nstep = max([int(nstep / record_num), 10]) * record_num
        self.cmd.append(
            self.DUMP_EVERY.format(id=self.DUMP_ID, arg=nstep * (num + 1)))
        # The variables defined here will be evaluated by ${xxx}
        self.cmd.append(self.SET_VOL)
        self.cmd.append(self.VARIABLE_AMP)
        self.cmd.append(self.SET_PRESS)
        self.cmd.append(self.SET_MODULUS % record_num)
        self.cmd.append(self.SET_FACTOR % self.options.press)
        # Start loop and cd into sub-dir as some files are of the same name
        loop_defm, defm_id, defm_break = 'loop_defm', 'defm_id', 'defm_break'
        self.cmd.append(f"variable {defm_id} loop 0 {max_loop - 1} pad")
        self.cmd.append(self.SET_LABEL % loop_defm)
        self.cmd.append('print "Deform Id  = ${defm_id}"')
        self.cmd.append("shell mkdir defm_${defm_id}")
        self.cmd.append("shell cd defm_${defm_id}\n")
        pre = self.getCyclePre(nstep, record_num=record_num)
        self.nvt(nstep=nstep * num, stemp=self.temp, temp=self.temp, pre=pre)
        self.cmd.append('print "Averaged Press = ${immed_press}"')
        self.cmd.append('print "Modulus = ${immed_modulus}"')
        self.cmd.append('print "Scale Factor  = ${factor}"\n')
        # If last loop or no scaling, break and record properties
        self.cmd.append(
            f'if "${{defm_id}} == {max_loop - 1} || ${{factor}} == 1" '
            f'then "jump SELF {defm_break}"\n')
        self.nvt(nstep=nstep / 2,
                 stemp=self.temp,
                 temp=self.temp,
                 pre=self.FIX_DEFORM)
        self.nvt(nstep=nstep / 2, stemp=self.temp, temp=self.temp)
        self.cmd.append("shell cd ..")
        self.cmd.append(f"next {defm_id}")
        self.cmd.append(f"jump SELF {loop_defm}\n")
        self.cmd.append(f'label {defm_break}')
        self.cmd.append('variable ave_press equal ${immed_press}')
        self.cmd.append('variable modulus equal ${immed_modulus}')
        self.cmd.append(
            self.DUMP_EVERY.format(id=self.DUMP_ID, arg=self.DUMP_Q))
        self.cmd.append('shell cd ..\n')

    def getCyclePre(self, nstep, record_num=100):
        """
        Get the pre-stage str for the cycle simulation.

        :param nstep int: the simulation steps of the one cycles
        :param record_num int: each cycle records this number of data
        :return str: the prefix string of the cycle stage.
        """
        params = ' '.join([self.WIGGLE_DIM % dim for dim in ['x', 'y', 'z']])
        period = nstep * self.timestep
        wiggle = self.WIGGLE_VOL.format(PARAM=params).format(period=period)
        record_period = int(nstep / record_num)
        record_press = self.RECORD_PRESS_VOL.format(period=record_period)
        return record_press + wiggle

    def relaxAndDefrom(self):
        """
        Longer relaxation at constant temperature and deform to the mean size.
        """
        if self.testing:
            return
        if self.options.prod_ens == NPT:
            self.npt(nstep=self.relax_step,
                     stemp=self.temp,
                     temp=self.temp,
                     press=self.press,
                     modulus="${modulus}")
            return
        # NVE and NVT production runs use averaged cell
        pre = self.getBdryPre()
        self.npt(nstep=self.relax_step,
                 stemp=self.temp,
                 temp=self.temp,
                 press=self.press,
                 modulus="${modulus}",
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
        if self.options.prod_ens == NVE:
            self.nve(nstep=self.prod_step)
        elif self.options.prod_ens == NVT:
            self.nvt(nstep=self.prod_step, stemp=self.temp, temp=self.temp)
        else:
            self.npt(nstep=self.prod_step,
                     stemp=self.temp,
                     temp=self.temp,
                     press=self.press,
                     modulus="${modulus}")

    def nve(self, nstep=1E3):
        """
        Append command for constant energy and volume.

        :nstep int: run this steps for time integration.
        """
        # NVT on single molecule gives nan coords (guess due to translation)
        cmd = self.FIX_NVE + self.RUN_STEP % nstep + self.UNFIX
        self.cmd.append(cmd)

    def nvt(self,
            nstep=1E4,
            stemp=300,
            temp=300,
            style=TEMP_BERENDSEN,
            pre=''):
        """
        Append command for constant volume and temperature.

        :nstep int: run this steps for time integration
        :stemp float: starting temperature
        :temp float: target temperature
        :style str: the style for the command
        :pre str: additional pre-conditions
        """
        if style == self.TEMP_BERENDSEN:
            cmd1 = self.FIX_TEMP_BERENDSEN.format(stemp=stemp,
                                                  temp=temp,
                                                  tdamp=self.tdamp)
            cmd2 = self.FIX_NVE
        cmd = pre + cmd1 + cmd2
        fx = [x for x in cmd.split(symbols.RETURN) if x.startswith(self.FIX)]
        self.cmd.append(cmd + self.RUN_STEP % nstep + self.UNFIX * len(fx))

    def npt(self,
            nstep=20000,
            stemp=300,
            temp=300,
            spress=1.,
            press=1.,
            style=PRESS_BERENDSEN,
            modulus=10,
            pdamp=None,
            pre=''):
        """
        Append command for constant pressure and temperature.

        :nstep int: run this steps for time integration
        :stemp int: starting temperature
        :temp float: target temperature
        :spress float: starting pressure
        :press float: target pressure
        :style str: the style for the command
        :pdamp pdamp: Pressure damping parameter (x timestep to get the param)
        :pre str: additional pre-conditions
        """
        if pdamp is None:
            pdamp = self.pdamp
        if spress is None:
            spress = press
        if style == self.PRESS_BERENDSEN:
            cmd1 = self.FIX_PRESS_BERENDSEN.format(spress=spress,
                                                   press=press,
                                                   pdamp=pdamp,
                                                   modulus=modulus)
            cmd2 = self.FIX_TEMP_BERENDSEN.format(stemp=stemp,
                                                  temp=temp,
                                                  tdamp=self.tdamp)
            cmd3 = self.FIX_NVE
        cmd = pre + cmd1 + cmd2 + cmd3
        fx = [x for x in cmd.split(symbols.RETURN) if x.startswith(self.FIX)]
        self.cmd.append(cmd + self.RUN_STEP % nstep + self.UNFIX * len(fx))

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

    def writeFixShake(self):
        """
        Write fix shake command to constrain bonds and angles. This method
        should be overwritten when force field and structure are available.
        """
        return

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

    def writeRun(self, mols=None, struct=None):
        """
        Write command to further equilibrate the system.

        :param mols dict: id and rdkit.Chem.rdchem.Mol
        """
        self.in_fh.write(f"velocity all create {self.options.stemp} 482748\n")
        fwriter = FixWriter(self.in_fh,
                            options=self.options,
                            mols=mols,
                            struct=struct)
        fwriter.run()
