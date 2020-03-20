#------------------------------------------------------------------------------#
# Set the minimum CMake version
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.2)
cmake_policy(SET CMP0074 NEW)
include(ExternalProject)

#------------------------------------------------------------------------------#
# Setup the project
#------------------------------------------------------------------------------#

project(CONTRA VERSION 0.1.0)

# We need C++ 14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)

# cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# project variables
set( CONTRA_TEST_TOLERANCE 1e-14 CACHE INTERNAL "")
set( CONTRA_TOOL_DIR "${PROJECT_SOURCE_DIR}/tools" CACHE INTERNAL "")

#------------------------------------------------------------------------------#
# configure targets
#------------------------------------------------------------------------------#

add_library( librt )
add_executable( contra )
target_link_libraries(contra PUBLIC librt)


#------------------------------------------------------------------------------#
# LLVM
#------------------------------------------------------------------------------#

find_package(LLVM REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

target_include_directories(contra PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(contra PUBLIC ${LLVM_DEFINITIONS})

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs core mcjit native orcjit objcarcopts)
set( llvm_libs LLVM )
target_link_libraries(contra PUBLIC ${llvm_libs})

#------------------------------------------------------------------------------#
# Legion
#------------------------------------------------------------------------------#

set(LEGION_DIR ${PROJECT_SOURCE_DIR}/legion)
file(GLOB _legion_contents ${LEGION_DIR}/*)
if (_legion_contents)
  ExternalProject_Add(legion
    SOURCE_DIR  ${LEGION_DIR}
    INSTALL_DIR  ${CMAKE_INSTALL_PREFIX}
    UPDATE_COMMAND  ""
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  )
  set( Legion_ROOT "${CMAKE_INSTALL_PREFIX}" CACHE FORCE "Legion root install directory.")
endif()

find_package(Legion)
if (Legion_FOUND)
  
  # need pthreads
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  # need zlib
  find_package(ZLIB REQUIRED)
  # add other necessary libaries for legion
  list(APPEND Legion_LIBRARIES Threads::Threads ${ZLIB_LIBRARIES} ${CMAKE_DL_LIBS})
  if(UNIX AND NOT APPLE)
    list(APPEND Legion_LIBRARIES rt)
  endif()
  target_include_directories(librt PUBLIC ${Legion_INCLUDE_DIRS})
  target_link_libraries(librt PUBLIC ${Legion_LIBRARIES})
endif()

#------------------------------------------------------------------------------#
# build library
#------------------------------------------------------------------------------#

target_include_directories(librt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src/librt)
add_subdirectory(src/contra)

#------------------------------------------------------------------------------#
# Enable Regression Tests
#------------------------------------------------------------------------------#
include(CTest) # note: this adds a BUILD_TESTING which defaults to ON

# find python for running regression tests
set(PythonInterp_FIND_VERSION TRUE) # unknown hack to search for versions
find_package (PythonInterp QUIET)

if (PYTHONINTERP_FOUND)
  message (STATUS "Found PythonInterp: ${PYTHON_EXECUTABLE}")
endif ()

if(BUILD_TESTING AND NOT PYTHONINTERP_FOUND)
  message(FATAL_ERROR "Testing requested, but python was not found")
endif()

#------------------------------------------------------------------------------#
# Add examples
#------------------------------------------------------------------------------#

if (BUILD_TESTING)
  include(test)
  add_subdirectory(examples)
endif()
