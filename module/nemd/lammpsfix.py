NVT = 'NVT'
NPT = 'NPT'
NVE = 'NVE'
ENSEMBLES = [NVE, NVT, NPT]

FIX = 'fix'
RUN_STEP = "run %i\n"
UNFIX = "unfix %s\n"

FIX_NVE = f"{FIX} %s all nve\n"
FIX_NVT = f"{FIX} %s all nvt temp {{stemp}} {{temp}} {{tdamp}}\n"

TEMP = 'temp'
BERENDSEN = 'berendsen'
TEMP_BERENDSEN = f'{TEMP}/{BERENDSEN}'
FIX_TEMP_BERENDSEN = f"{FIX} %s all {TEMP_BERENDSEN} {{stemp}} {{temp}} {{tdamp}}\n"

DUMP_EVERY = "dump_modify {id} every {arg}"
DUMP_ID, DUMP_Q = 1, 1000

PRESS = 'press'
PRESS_BERENDSEN = f'{PRESS}/{BERENDSEN}'
MODULUS = 'modulus'
FIX_PRESS_BERENDSEN = f"{FIX} %s all {PRESS_BERENDSEN} iso {{spress}} {{press}} {{pdamp}} {MODULUS} {{modulus}}\n"

CHANGE_BOX = "change_box all x scale ${factor} y scale ${factor} z scale ${factor} remap\n"
DEFORM = 'deform'
FIX_DEFORM = f"{FIX} %s all {DEFORM} 100 x scale ${{factor}} y scale ${{factor}} z scale ${{factor}} remap v\n"
AMP = 'amp'
VOL = 'vol'
SET_AMP = f'variable {AMP} equal 0.01*{VOL}^(1/3)\n'
WIGGLE_DIM = "%s wiggle ${{amp}} {period}"
WIGGLE_VOL = f"{FIX} %s all {DEFORM} 100 {{PARAM}}\n"

SET_VOL = f"variable {VOL} equal {VOL}"
PRESS_VOL_FILE = 'press_vol.data'
RECORD_PRESS_VOL = f"{FIX} %s all ave/time 1 {{period}} {{period}} " \
                   f"c_thermo_{PRESS} v_{VOL} file {PRESS_VOL_FILE}\n"

IMMED_MODULUS = 'immed_modulus'
SET_MODULUS = f"""variable {IMMED_MODULUS} python getModulus
python getModulus input 2 {PRESS_VOL_FILE} {{record_num}} return v_{IMMED_MODULUS} format sif here "from nemd.pyfunc import getModulus"
"""

IMMED_PRESS = 'immed_press'
SET_PRESS = f"""variable {IMMED_PRESS} python getPress
python getPress input 1 {PRESS_VOL_FILE} return v_{IMMED_PRESS} format sf here "from nemd.pyfunc import getPress"
"""

FACTOR = 'factor'
SET_FACTOR = f"""variable {FACTOR} python getBdryFactor
python getBdryFactor input 2 {{press}} press_vol.data return v_{FACTOR} format fsf here "from nemd.pyfunc import getBdryFactor"
"""

XYZL_FILE = 'xyzl.data'
RECORD_BDRY = f"""
variable xl equal "xhi - xlo"
variable yl equal "yhi - ylo"
variable zl equal "zhi - zlo"
fix %s all ave/time 1 1000 1000 v_xl v_yl v_zl file {XYZL_FILE}
"""

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