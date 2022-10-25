#     units real
#     atom_style full
#     bond_style harmonic
#     angle_style harmonic
#     dihedral_style opls
#     improper_style harmonic
#     pair_style lj/cut/coul/long 11.0 11.0
#       # (Note to self: The pair_style used by OPLSAA/M (2015) is:
#       #                lj/charmm/coul/long 9.0 11.0
#       #                ...so this will have to be updated eventually.)
#     pair_modify mix geometric
#     special_bonds lj/coul 0.0 0.0 0.5
#     kspace_style pppm 0.0001