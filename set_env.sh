#!/bin/bash

unset ADIOS2_DIR


clean_path() {
    echo "$1" | tr ':' '\n' | grep -v '/home/adios/local/adios-install' | grep -v '/opt/adios2' | paste -sd:
}

export PATH=$(clean_path "$PATH")
export LD_LIBRARY_PATH=$(clean_path "$LD_LIBRARY_PATH")
export LIBRARY_PATH=$(clean_path "$LIBRARY_PATH")
export CPATH=$(clean_path "$CPATH")
export PKG_CONFIG_PATH=$(clean_path "$PKG_CONFIG_PATH")
export MANPATH=$(clean_path "$MANPATH")


export ADIOS2_DIR=/home/adios/Programs/ADIOS2/install
export PATH=$ADIOS2_DIR/bin:$PATH
export LD_LIBRARY_PATH=$ADIOS2_DIR/lib:/home/adios/Programs/CAESAR_C/install/lib:/home/adios/local/MGARD_install/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ADIOS2_DIR/lib:$LIBRARY_PATH
export CPATH=$ADIOS2_DIR/include:$CPATH
export PKG_CONFIG_PATH=$ADIOS2_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export MANPATH=$ADIOS2_DIR/share/man:$MANPATH


echo "Environment set to use:"
echo "   ADIOS2_DIR       = $ADIOS2_DIR"
echo "   PATH             = $(echo $PATH | tr ':' '\n' | grep ADIOS2)"
echo "   LD_LIBRARY_PATH  = $LD_LIBRARY_PATH"
echo "   LIBRARY_PATH     = $LIBRARY_PATH"
echo "   CPATH            = $CPATH"
echo "   PKG_CONFIG_PATH  = $PKG_CONFIG_PATH"
echo "   MANPATH          = $MANPATH"

