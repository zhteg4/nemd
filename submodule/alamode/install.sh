#! /bin/zsh

git submodule add https://github.com/ttadano/alamode.git
git -C alamode checkout develop

mkdir build; cd build
cmake -DUSE_MKL_FFT=yes -DFFTW3_ROOT=/usr/local -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which g++` -DSPGLIB_ROOT=`pip3 show spglib | grep 'Location:' | sed 's/^.*: //'`/spglib ../alamode
make

