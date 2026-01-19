#!/bin/bash
# =====================================================================
# UF HiPerGator ADIOS2 + CAESAR + CUDA + nvCOMP + Torch environment
# =====================================================================

# --------------------------
# Step 0: Clean old paths
# --------------------------
clean_path() {
    echo "$1" | tr ':' '\n' | \
        grep -v -E 'ADIOS|adios|CAESAR|caesar|MGARD|mgard|nvcomp|cuda|CUDA|Torch|torch|mpich|mpi' | \
        paste -sd:
}

export PATH=$(clean_path "$PATH")
export LD_LIBRARY_PATH=$(clean_path "$LD_LIBRARY_PATH")
export LIBRARY_PATH=$(clean_path "$LIBRARY_PATH")
export CPATH=$(clean_path "$CPATH")
export PKG_CONFIG_PATH=$(clean_path "$PKG_CONFIG_PATH")
export MANPATH=$(clean_path "$MANPATH")

# --------------------------
# Step 1: Load CUDA module
# --------------------------
module load cuda/12.8.1
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH

echo "→ CUDA loaded from $CUDA_HOME"
which nvcc

# --------------------------
# Step 2: ADIOS2 and CAESAR
# --------------------------
export ADIOS2_DIR=/lustre/blue/ranka/eklasky/ADIOS2/install
export CAESAR_DIR=/lustre/blue/ranka/eklasky/CAESAR_C/install

export PATH=$ADIOS2_DIR/bin:$CAESAR_DIR/bin:$PATH
export LD_LIBRARY_PATH=$ADIOS2_DIR/lib64:$CAESAR_DIR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ADIOS2_DIR/lib64:$CAESAR_DIR/lib:$LIBRARY_PATH
export CPATH=$ADIOS2_DIR/include:$CAESAR_DIR/include:$CPATH
export PKG_CONFIG_PATH=$ADIOS2_DIR/lib64/pkgconfig:$CAESAR_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export MANPATH=$ADIOS2_DIR/share/man:$MANPATH

# --------------------------
# Step 3: MGARD
# --------------------------
export MGARD_DIR=/home/eklasky/Software/MGARD/build_scripts/install-serial
export PATH=$MGARD_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MGARD_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$MGARD_DIR/lib64:$LIBRARY_PATH
export CPATH=$MGARD_DIR/include:$CPATH
export PKG_CONFIG_PATH=$MGARD_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/eklasky/Software/MGARD/build_scripts/install-serial/lib64:$LD_LIBRARY_PATH


# --------------------------
# Step 4: nvCOMP
# --------------------------
export NVCOMP_DIR=$HOME/local/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive
export PATH=$NVCOMP_DIR/bin:$PATH
export LD_LIBRARY_PATH=$NVCOMP_DIR/lib:$LD_LIBRARY_PATH
export CPATH=$NVCOMP_DIR/include:$CPATH
export CMAKE_PREFIX_PATH=$NVCOMP_DIR:$CMAKE_PREFIX_PATH

# --------------------------
# Step 5: libTorch / PyTorch
# --------------------------
export Torch_DIR=/lustre/blue/ranka/eklasky/caesar_venv/lib/python3.11/site-packages/torch/share/cmake/Torch
export CMAKE_PREFIX_PATH=$Torch_DIR:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/lustre/blue/ranka/eklasky/caesar_venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

# --------------------------
export MPICH_DIR=$HOME/local/mpich-4.3.1

export PATH=$MPICH_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPICH_DIR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$MPICH_DIR/lib:$LIBRARY_PATH
export CPATH=$MPICH_DIR/include:$CPATH
export PKG_CONFIG_PATH=$MPICH_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export PYTHONPATH=/lustre/blue/ranka/eklasky/ADIOS2/build/blue/ranka/eklasky/caesar_venv/lib/python3.11/site-packages:$PYTHONPATH


export LD_LIBRARY_PATH=/home/eklasky/Software/MGARD/install-serial/lib64:$LD_LIBRARY_PATH

echo "→ MPICH loaded from $MPICH_DIR"
which mpicc
which mpirun

export UCX_TLS=tcp
export UCX_NET_DEVICES=lo

# --------------------------
# Step 6: Summary
# --------------------------
echo "============================================================"
echo "Environment set CLEANLY for ADIOS2 + CAESAR + CUDA + nvCOMP + Torch"
echo "------------------------------------------------------------"
echo "ADIOS2_DIR       = $ADIOS2_DIR"
echo "CAESAR_DIR       = $CAESAR_DIR"
echo "MGARD_DIR        = $MGARD_DIR"
echo "NVCOMP_DIR       = $NVCOMP_DIR"
echo "Torch_DIR        = $Torch_DIR"
echo
echo "PATH:"
echo "$PATH" | tr ':' '\n'
echo
echo "LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n'
echo "============================================================"

