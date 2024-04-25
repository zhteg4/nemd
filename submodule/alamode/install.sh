#! /bin/zsh

# git submodule add -b develop https://github.com/ttadano/alamode.git
git submodule update --init --recursive
mkdir build; cd build
cmake -DFFTW3_ROOT=/usr/local -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which clang++ || which g++` -DSPGLIB_ROOT=`pip3 show spglib | grep 'Location:' | sed 's/^.*: //'`/spglib ../alamode
make

