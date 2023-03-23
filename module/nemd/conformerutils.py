# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles 'rdkit.Chem.rdchem.Conformer' for translation, rotation,
centroid, and so on.
"""
import numpy as np
from rdkit import rdBase
from rdkit import DataStructs
from rdkit import Chem
from scipy.spatial.transform import Rotation


def translation(conf, vect):
    """
    Do translation on this conformer using this vector.

    :param conf 'rdkit.Chem.rdchem.Conformer': the conformer to do translation on.
    :param vect 'numpy.ndarray': translational vector
    """
    mtrx = np.identity(4)
    mtrx[:-1, 3] = vect
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)


def rotate(conf, ivect, tvect):
    """
    Rotate the conformer by three initial vectors and three target vectors.

    :param conf 'rdkit.Chem.rdchem.Conformer': rotate this conformer
    :param ivect 3x3 'numpy.ndarray': Each row is one initial vector
    :param tvect 3x3 'numpy.ndarray': Each row is one corresponding target vector
    """
    rotation, rmsd = Rotation.align_vectors(tvect, ivect)
    mtrx = np.identity(4)
    mtrx[:-1, :-1] = rotation.as_matrix()
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)


def rand_rotate(conf):
    """
    Randomly rotate the conformer.

    :param conf 'rdkit.Chem.rdchem.Conformer': rotate this conformer

    NOTE: the random state is set according to the numpy random seed.
    """
    mtrx = np.identity(4)
    seed = np.random.randint(0, 2**32 - 1)
    mtrx[:-1, :-1] = Rotation.random(random_state=seed).as_matrix()
    Chem.rdMolTransforms.TransformConformer(conf, mtrx)


def centroid(conf, aids=None):
    """
    Compute the centroid of the whole conformer ar the selected atoms.

    :param conf 'rdkit.Chem.rdchem.Conformer': whose centroid to be computed
    :param atom_ids list: the selected atom ids
    """
    if aids is None:
        return Chem.rdMolTransforms.ComputeCentroid(conf)

    bv = DataStructs.ExplicitBitVect(conf.GetNumAtoms())
    bv.SetBitsFromList(aids)
    weights = rdBase._vectd()
    weights.extend(bv.ToList())
    return Chem.rdMolTransforms.ComputeCentroid(conf,
                                                weights=weights,
                                                ignoreHs=False)
