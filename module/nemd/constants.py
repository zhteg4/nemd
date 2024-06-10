# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module contains shared numerical constants.
"""
from scipy import constants

LARGE_NUM = 1000000
ONE_ONE_ONE = (1, 1, 1,) # yapf: disable
ANG_TO_CM = constants.angstrom / constants.centi
CM_INV_THZ = constants.physical_constants['inverse meter-hertz relationship'][
    0] / constants.tera / constants.centi
