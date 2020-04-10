Contra
======



Installation
----------------


    # Build third-party libraries (optional)
    cd external
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$PWD/install ..
    make -j
    cd ../../
        
    # Build contra
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=$PWD/../external/build/install -DCMAKE_INSTALL_PREFIX=$PWD/install ..
        

Getting Started
---------------

Documentation
---------------