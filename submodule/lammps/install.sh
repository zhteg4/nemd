#! /bin/zsh

brew install clang-format

git submodule add -b develop https://github.com/lammps/lammps.git

mkdir build; cd build
cmake -DPKG_PYTHON=yes -DPKG_MOLECULE=yes -DPKG_KSPACE=yes -DPKG_RIGID=yes -S ../lammps/cmake/ -B .
cmake --build . --config Release -j4
make install
