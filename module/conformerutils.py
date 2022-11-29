import numpy as np
from rdkit import Chem

def translation(conf, vect):
    mtrx = np.identity(4)
    mtrx[:-1, 3] = vect
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)