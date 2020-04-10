#------------------------------------------------------------------------------#
# Set the minimum CMake version
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.2)
cmake_policy(SET CMP0074 NEW)
include(ExternalProject)

#------------------------------------------------------------------------------#
# Setup the project
#------------------------------------------------------------------------------#

project(CONTRA-TPL VERSION 0.1.0)

# cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

#------------------------------------------------------------------------------#
# Legion
#------------------------------------------------------------------------------#
option(BUILD_LLVM "Build llvm." ON)
if (BUILD_LLVM)
  MESSAGE(STATUS "Building LLVM.")
  ExternalProject_Add(llvm
    URL https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/llvm-9.0.1.src.tar.xz
    INSTALL_DIR  ${CMAKE_INSTALL_PREFIX}
    UPDATE_COMMAND  ""
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DLLVM_ENABLE_RTTI=ON
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
  )
else()
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
endif()

#------------------------------------------------------------------------------#
# Legion
#------------------------------------------------------------------------------#
option(BUILD_LEGION "Build legion." ON)
if (BUILD_LEGION)
  MESSAGE(STATUS "Building Legion.")
  ExternalProject_Add(legion
    GIT_REPOSITORY http://github.com/StanfordLegion/legion.git
    INSTALL_DIR  ${CMAKE_INSTALL_PREFIX}
    UPDATE_COMMAND  ""
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DBUILD_SHARED_LIBS=True
  )
else()
  MESSAGE(STATUS "Found Legion")
  MESSAGE(STATUS "Legion includes: ${Legion_INCLUDE_DIRS}")
  MESSAGE(STATUS "Legion libraries: ${Legion_LIBRARIES}")
endif()

