# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module contains shared non-numerical symbols.
"""
WILD_CARD = '*'
CARBON = 'C'
UNKNOWN = 'X'
HYDROGEN = 'H'
NITROGEN = 'N'
OXYGEN = 'O'
POUND = '#'
SEP = f'_{POUND}_'
BACKSLASH = '\\'
FORWARDSLASH = '/'
RETURN = '\n'
RC_BRACKET = '}'
DOUBLE_QUOTATION = '"'
SEMICOLON = ';'
COMMA = ','
PERIOD = '.'
AND = '&'
PLUS_MIN = '\u00B1'
ELEMENT_OF = '\u2208'
ANGSTROM = '\u212B'
XU = 'xu'
YU = 'yu'
ZU = 'zu'
XYZU = [XU, YU, ZU]
SPC = 'SPC'
SPCE = 'SPCE'
TIP3P = 'TIP3P'
WMODELS = [SPC, SPCE, TIP3P]
IMPLICIT_H = 'implicit_h'
OPLSUA = 'OPLSUA'
FF_MODEL = {OPLSUA: WMODELS}
OPLSUA_TIP3P = f'{OPLSUA},{TIP3P}'
SPACE = ' '
SPACE_PATTERN = r'\s+'
TIME_LB = 'Time ({unit})'
FS = 'fs'
PS = 'ps'
