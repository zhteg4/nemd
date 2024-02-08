#! /bin/zsh
brew install boost
brew install eigen
brew install spglib
brew install fftw
brew install cmake
git clone https://github.com/ttadano/alamode.git
cd alamode
git checkout develop
mkdir _build; cd _build
cmake -DUSE_MKL_FFT=yes -DSPGLIB_ROOT=${SPGLIB_ROOT} \
    -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS="-O2 -xHOST" ..

