#! /bin/zsh

git submodule add https://github.com/ttadano/alamode.git
git -C alamode checkout develop

export CC=/usr/local/opt/llvm/bin/clang
export SPGLIB_ROOT=/usr/local/Cellar/spglib/2.3.0
export LD_LIBRARY_PATH=$SPGLIB_ROOT/lib:$LD_LIBRARY_PATH

mkdir build; cd build
cmake -DUSE_MKL_FFT=yes -DSPGLIB_ROOT=/usr/local -DFFTW3_ROOT=/usr/local -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++  ../alamode
make

