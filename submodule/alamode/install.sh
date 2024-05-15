#! /bin/zsh

# git submodule add -b develop https://github.com/ttadano/alamode.git
git submodule update --init --recursive
mkdir build; cd build

FFTW3_ROOT=/usr/local

if command -v brew &> /dev/null
then
  SPGLIB_ROOT=$(echo $(brew --cellar spglib)/*/lib* | sed 's/lib*$//')
  LLVM_PATH=$(brew info llvm | grep 'export PATH' | sed 's/^.*export PATH=//; s/:$PATH.*$//; s/\"//')
  [[ ":$PATH:" != *":$LLVM_PATH:"* ]] && export PATH="$LLVM_PATH${PATH:+":$PATH"}"
  CMAKE_C_COMPILER=$(which clang)
  CMAKE_CXX_COMPILER=$(which clang++)
else
  SPGLIB_ROOT=`pip3 show spglib | grep 'Location:' | sed 's/^.*: //'`/spglib
  CMAKE_C_COMPILER=$(which gcc)
  CMAKE_CXX_COMPILER=$(which g++)
fi

set -x;
cmake -DCMAKE_C_COMPILER=$CMAKE_C_COMPILER -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER -DSPGLIB_ROOT=$SPGLIB_ROOT -DFFTW3_ROOT=$FFTW3_ROOT ../alamode
set +x;
make

