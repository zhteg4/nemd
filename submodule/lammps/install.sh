#! /bin/zsh

brew install clang-format

git submodule add -b develop https://github.com/lammps/lammps.git

mkdir build; cd build
cmake -D PKG_PYTHON=on -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on -D BUILD_LIB=on -D BUILD_SHARED_LIBS=yes -D PYTHON_EXECUTABLE="/usr/local/bin/python3" -D CMAKE_INSTALL_PREFIX=/usr/local -D LAMMPS_MACHINE=serial -S ../lammps/cmake/ -B .
cmake --build . -j4
make install-python