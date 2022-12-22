from rdkit import Chem
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