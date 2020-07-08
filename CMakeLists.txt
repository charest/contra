#------------------------------------------------------------------------------#
# Set the minimum CMake version
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.2)
cmake_policy(SET CMP0074 NEW)
include(ExternalProject)

#------------------------------------------------------------------------------#
# Setup the project
#------------------------------------------------------------------------------#

project(Contra VERSION 0.1.0)

# We need C++ 14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)

# cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
include(BuildType)

# project variables
set( CONTRA_TEST_TOLERANCE 1e-14 CACHE INTERNAL "")
set( CONTRA_TOOL_DIR "${PROJECT_SOURCE_DIR}/tools" CACHE INTERNAL "")

# options
option(BUILD_SHARED_LIBS "Build shared libraries." ON)

set(SUPPORTED_BACKENDS "serial")

#------------------------------------------------------------------------------#
# configure targets
#------------------------------------------------------------------------------#

add_executable( contra )


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
if (BUILD_SHARED_LIBS)
  llvm_config(contra USE_SHARED all)
else()
  llvm_config(contra all)
endif()

#------------------------------------------------------------------------------#
# Legion
#------------------------------------------------------------------------------#

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Legion)
if (Legion_FOUND)
  MESSAGE(STATUS "Legion includes: ${Legion_INCLUDE_DIRS}")
  MESSAGE(STATUS "Legion libraries: ${Legion_LIBRARIES}")
  
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
  target_include_directories(contra PUBLIC ${Legion_INCLUDE_DIRS})
  target_link_libraries(contra PUBLIC ${Legion_LIBRARIES})

  list(APPEND SUPPORTED_BACKENDS "legion")
endif()

#------------------------------------------------------------------------------#
# Kokkos
#------------------------------------------------------------------------------#
find_package(Kokkos)
if (Kokkos_FOUND)
  target_link_libraries(contra PUBLIC Kokkos::kokkos)
  #list(APPEND SUPPORTED_BACKENDS "kokkos")
endif()

#------------------------------------------------------------------------------#
# Cuda toolkit
#------------------------------------------------------------------------------#
find_package(CUDA)
if (CUDA_FOUND)
  enable_language(CUDA)
  
  add_library( contra_cuda STATIC )
  target_compile_options(contra_cuda PUBLIC -lib -rdc=true )
  target_include_directories(contra_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_include_directories(contra_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
  
  target_link_libraries(contra PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  #target_link_libraries(contra PUBLIC /projects/opt/centos7/cuda/10.2/lib64/libnvrtc-builtins.so.10.2)
  #target_link_libraries(contra PUBLIC /projects/opt/centos7/cuda/10.2/lib64/libnvrtc.so.10.2.89)
  target_include_directories(contra PUBLIC ${CUDA_INCLUDE_DIRS})
  
  list(APPEND SUPPORTED_BACKENDS "cuda")
endif()

#------------------------------------------------------------------------------#
# build library
#------------------------------------------------------------------------------#

MESSAGE(STATUS "Supported backends: ${SUPPORTED_BACKENDS}")

target_include_directories(contra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(contra PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)

add_subdirectory(src)

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
  add_subdirectory(testing)
  add_subdirectory(examples)
endif()




#------------------------------------------------------------------------------#
# Export targets and package.
#------------------------------------------------------------------------------#

install( TARGETS contra contra_cuda
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include) 

file(GLOB_RECURSE test_files testing "*.cta")
file(GLOB_RECURSE sample_files examples/contra "*.cta")
install( FILES ${test_files} ${sample_files} DESTINATION share/examples )


install( EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE Contra::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_file(${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
  ${PROJECT_NAME}Config.cmake @ONLY)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)
