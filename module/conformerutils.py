import numpy as np
from rdkit import rdBase
from rdkit import DataStructs
from rdkit import Chem
from scipy.spatial.transform import Rotation


def translation(conf, vect):
    mtrx = np.identity(4)
    mtrx[:-1, 3] = vect
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)

def rotate(conf, ivect, tvect):
    rotation, rmsd = Rotation.align_vectors(tvect, ivect)
    mtrx = np.identity(4)
    mtrx[:-1, :-1] = rotation.as_matrix()
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)

def rand_rotate(conf):
    mtrx = np.identity(4)
    mtrx[:-1, :-1] = Rotation.random().as_matrix()
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)

def centroid(conf, atom_ids=None):
    if atom_ids is None:
        return Chem.rdMolTransforms.ComputeCentroid(conf)

    bv = DataStructs.ExplicitBitVect(conf.GetNumAtoms())
    bv.SetBitsFromList(atom_ids)
    weights = rdBase._vectd().extend(bv.ToList())
    return Chem.rdMolTransforms.ComputeCentroid(conf, weights=weights)
