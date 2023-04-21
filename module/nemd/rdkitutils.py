# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module provides utilities for `rdkit`.
"""
import os
import sys
import rdkit
import logging
from rdkit import Chem
from rdkit import rdBase
from rdkit import RDLogger
from rdkit.Chem import AllChem
from contextlib import contextmanager


@contextmanager
def rdkit_preserve_hs():
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    try:
        yield ps
    finally:
        ps.removeHs = True


@contextmanager
def rdkit_warnings_ignored():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    try:
        yield lg
    finally:
        lg.setLevel(RDLogger.WARNING)


def get_mol_from_smiles(smiles_str, embeded=True):
    with rdkit_preserve_hs() as ps:
        mol = Chem.MolFromSmiles(smiles_str, ps)
    if not embeded:
        return mol
    with rdkit_warnings_ignored():
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    return mol


class CaptureLogger(logging.Handler):
    """Helper class that captures Python logger output"""
    def __init__(self):
        super(CaptureLogger, self).__init__(level=logging.DEBUG)
        self.logs = {}
        self.devnull = open(os.devnull, 'w')
        rdkit.log_handler.setStream(self.devnull)
        rdkit.logger.addHandler(self)

    def __enter__(self):
        rdBase.LogToPythonLogger()
        return self.logs

    def __exit__(self, *args):
        self.release()
        rdBase.LogToCppStreams()

    def handle(self, record):
        key = record.levelname
        val = self.format(record)
        self.logs[key] = self.logs.get(key, "") + val
        return False

    def release(self):
        rdkit.log_handler.setStream(sys.stderr)
        rdkit.logger.removeHandler(self)
        self.devnull.close()
        return self.logs
