#! /bin/zsh

# git submodule add -b develop https://github.com/lammps/lammps.git
git submodule update --init --recursive
mkdir build; cd build
# which python3 -> /usr/bin/python3 in Linux but /usr/local/bin in Mac
cmake -D PKG_PYTHON=on -D PKG_MANYBODY=on -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on -D BUILD_LIB=on -D BUILD_SHARED_LIBS=yes -D PYTHON_EXECUTABLE=$(which python3) -D CMAKE_INSTALL_PREFIX=/usr/local -D LAMMPS_MACHINE=serial $@ -S ../lammps/cmake/ -B .
cmake --build . -j4
make install-python
