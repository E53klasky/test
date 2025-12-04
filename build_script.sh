#!/bin/bash

set -e
set -x

######## User Configurations ########
# Source directory
test_src_dir=.
# Build directory
test_build_dir=./build

export CMAKE_PREFIX_PATH=$HOME/Software/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/Software/nvcomp/lib:$HOME/Software/miniforge3/envs/py311torch/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
# export PKG_CONFIG_PATH=~/Software/miniforge3/envs/py311torch/lib/pkgconfig:$PKG_CONFIG_PATH

mkdir -p ${test_build_dir}

cmake -S ${test_src_dir} -B ${test_build_dir} \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

cmake --build ${test_build_dir} -- -j$(nproc)