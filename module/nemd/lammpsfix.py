SET_VAR = "variable {var} equal {expr}"

DUMP_EVERY = "dump_modify {id} every {arg}\n"
DUMP_ID, DUMP_Q = 1, 1000

FIX = 'fix'
RUN_STEP = "run %i\n"
UNFIX = "unfix %s\n"

NVT = 'NVT'
NPT = 'NPT'
NVE = 'NVE'
ENSEMBLES = [NVE, NVT, NPT]

FIX_NVE = f"{FIX} %s all nve\n"
FIX_NVT = f"{FIX} %s all nvt temp {{stemp}} {{temp}} {{tdamp}}\n"

TEMP = 'temp'
BERENDSEN = 'berendsen'
TEMP_BERENDSEN = f'{TEMP}/{BERENDSEN}'
FIX_TEMP_BERENDSEN = f"{FIX} %s all {TEMP_BERENDSEN} {{stemp}} {{temp}} {{tdamp}}\n"

PRESS = 'press'
PRESS_BERENDSEN = f'{PRESS}/{BERENDSEN}'
MODULUS = 'modulus'
FIX_PRESS_BERENDSEN = f"{FIX} %s all {PRESS_BERENDSEN} iso {{spress}} {{press}} {{pdamp}} {MODULUS} {{modulus}}\n"

VOL = 'vol'
PRESS_VOL_FILE = 'press_vol.data'
RECORD_PRESS_VOL = f"{FIX} %s all ave/time 1 {{period}} {{period}} c_thermo_{PRESS} v_{VOL} file {PRESS_VOL_FILE}\n"

IMMED_MODULUS = 'immed_modulus'
SET_IMMED_MODULUS = f"""variable {IMMED_MODULUS} python getModulus
python getModulus input 2 {PRESS_VOL_FILE} {{record_num}} return v_{IMMED_MODULUS} format sif here "from nemd.pyfunc import getModulus"
"""
SET_MODULUS = SET_VAR.format(var='modulus', expr=f'${{{IMMED_MODULUS}}}')

IMMED_PRESS = 'immed_press'
SET_IMMED_PRESS = f"""variable {IMMED_PRESS} python getPress
python getPress input 1 {PRESS_VOL_FILE} return v_{IMMED_PRESS} format sf here "from nemd.pyfunc import getPress"
"""
SET_PRESS = SET_VAR.format(var='press', expr=f'${{{IMMED_PRESS}}}')

FACTOR = 'factor'
SET_FACTOR = f"""variable {FACTOR} python getBdryFactor
python getBdryFactor input 2 {{press}} press_vol.data return v_{FACTOR} format fsf here "from nemd.pyfunc import getBdryFactor"
"""

DEFORM = 'deform'
FIX_DEFORM = f"{FIX} %s all {DEFORM} 100 x scale ${{{FACTOR}}} y scale ${{{FACTOR}}} z scale ${{{FACTOR}}} remap v\n"

AMP = 'amp'
WIGGLE_DIM = "%s wiggle ${{amp}} {period}"
PARAM = ' '.join([WIGGLE_DIM % dim for dim in ['x', 'y', 'z']])
WIGGLE_VOL = f"{FIX} %s all {DEFORM} 100 {PARAM}\n"

XYZL_FILE = 'xyzl.data'
RECORD_BDRY = f"""
variable xl equal "xhi - xlo"
variable yl equal "yhi - ylo"
variable zl equal "zhi - zlo"
fix %s all ave/time 1 1000 1000 v_xl v_yl v_zl file {XYZL_FILE}
"""

DEL_VAR = "variable {var} delete"

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

SET_LABEL = "label {label}"
SET_LOOP = "variable {id} loop 0 {end} pad"
MKDIR = "shell mkdir {dir}"
CD = "shell cd {dir}"
JUMP = "jump SELF {label}"
IF_JUMP = f'if "{{cond}}" then "{JUMP}"'
PRINT = 'print "{var} = ${{{var}}}"'
NEXT = "next {id}"
