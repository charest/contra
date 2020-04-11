Contra
======



Installation
----------------

Contra currently depends on [Legion](https://legion.stanford.edu/) and 
[LLVM](http://llvm.org/) version 9.  If you build Legion yourself, it *MUST* be
build as a shared library (i.e. with -DBUILD_SHARED_LIBS=on).

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

To run one of the examples,

    ./contra ../examples/contra/00_hello_world/hello.cta

will produce

    Hello World!
    
There are a bunch of tests and examples that you can try.  They are located in
the `examples` and `testing` folders.

Documentation
---------------

See the [wiki](https://gitlab.lanl.gov/charest/contra/-/wikis/home) for more
information on the Contra language and the compiler/interpreter. 