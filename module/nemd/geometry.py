import numpy as np


def distace(xyzs):
    return np.linalg.norm(xyzs[0] - xyzs[1])


def angle_vs(v1, v2):
    return np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) /
                     np.linalg.norm(v2)) / np.pi * 180.


def angle(xyzs):
    v1 = xyzs[0] - xyzs[1]
    v2 = xyzs[2] - xyzs[1]
    return angle_vs(v1, v2)


def dihedral(xyzs):
    n1 = np.cross(xyzs[0] - xyzs[1], xyzs[1] - xyzs[2])
    n2 = np.cross(xyzs[1] - xyzs[2], xyzs[2] - xyzs[3])
    return angle_vs(n1, n2)
