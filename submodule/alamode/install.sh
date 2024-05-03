#! /bin/zsh

# git submodule add -b develop https://github.com/ttadano/alamode.git
git submodule update --init --recursive
mkdir build; cd build

if command -v brew &> /dev/null
then
  LLVM_PATH=$(brew info llvm | grep 'export PATH' | sed 's/^.*export PATH=//; s/:$PATH.*$//; s/\"//')
  [[ ":$PATH:" != *":$LLVM_PATH:"* ]] && export PATH="$LLVM_PATH${PATH:+":$PATH"}"
fi

cmake -DFFTW3_ROOT=/usr/local -DCMAKE_C_COMPILER=`which clang || which gcc` -DCMAKE_CXX_COMPILER=`which clang++ || which g++` -DSPGLIB_ROOT=`pip3 show spglib | grep 'Location:' | sed 's/^.*: //'`/spglib ../alamode
make

