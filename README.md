What is Contra?
===============

Contra is a new programming language aimed at making parallel computing easy and portable.

Why Should I Use Contra?
------------------------

Because you are sick of writing and re-writing your code to take advantage of the ever-changing HPC landscape.  Write your complicated scientific code once, and harness the power of different parallel paradigms with the flip of a switch!

Do I Really Have to Learn a New Language?
-----------------------------------------

I'm sorry, but the Contra language is really, really simple.  I promise!  Although it is probably possible to modify an existing language like Fortran or C++, it's just to hard to ensure that you don't do something that breaks the programming model.  We get that its hard to keep track of all the different programming languages, so we strive to make learning Contra as simple as possible.

I Have All this Code Written in *My-Favorite-Language*, Can I Still Use It?
---------------------------------------------------------------------------

Yes.  We provide all the features of parallel paradigms like CUDA, MPI, Legion, etc.., and they themselves are libraries written in other languages.  It is definitely possible to provide access to your favorite library.

This is Amazing! How Do I Install It? 
-------------------------------------

Contra currently depends on [Legion](https://legion.stanford.edu/) and 
[LLVM](http://llvm.org/) version 9.  If you build Legion yourself, it *MUST* be
build as a shared library (i.e. with -DBUILD_SHARED_LIBS=on).

Building LLVM requires considerable resources.  The simplest approach is to install
pre-build binaries via your Linux package manager.  For example, on Ubuntu

    sudo apt install llvm-9-dev clang-9 libclang-9-dev

If prebuilt packages are not available, you can build them with CMake.

    # Build third-party libraries (optional)
    cd external
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$PWD/install -DBUILD_LEGION=on -DBUILD_LLVM=on ..
    make -j
    cd ../../

If you use pre-built binarys for LLVM, make sure to disable it when building the other
dependencies with `-DBUILD_LLVM=off`.  Building Contra is simple,
        
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

**Contra may be used as both a compiler and an interpreter.  However, it is recommended to use it as an interpreter in the manner shown above since Contra is still in early stages of development.**

Documentation
---------------

See the [wiki](docs/home.md) for more
information on the Contra language and the compiler/interpreter. 
