#! /bin/zsh
brew install gcc
brew install lapack
brew install open-mpi
brew install libomp
brew install boost
brew install eigen
brew install spglib
brew install fftw
brew install cmake
brew install llvm
git submodule add https://github.com/ttadano/alamode.git
cd alamode
git checkout develop

export CC=/usr/local/opt/llvm/bin/clang
export SPGLIB_ROOT=/usr/local/Cellar/spglib/2.3.0
export LD_LIBRARY_PATH=$SPGLIB_ROOT/lib:$LD_LIBRARY_PATH

cd alm
mkdir build; cd build
cmake -DSPGLIB_ROOT=/usr/local/Cellar/spglib/2.3.0 -S ..
make

cmake -DUSE_MKL_FFT=yes -DSPGLIB_ROOT=/usr/local -DFFTW3_ROOT=/usr/local -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++  ..

