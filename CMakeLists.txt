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
install( TARGETS contra
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION bin)

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
IF (CMAKE_VERSION VERSION_GREATER_EQUAL "3.17.0")
  find_package(CUDAToolkit)
  set(CUDA_FOUND ${CUDAToolkit_FOUND})
else()
  find_package(CUDA)
endif()


if (CUDA_FOUND)
  enable_language(CUDA)
  
  add_library( contra_cuda_rt STATIC )
  target_compile_options(contra_cuda_rt PRIVATE -rdc=true )
  target_include_directories(contra_cuda_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_include_directories(contra_cuda_rt PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
  
  add_library( rtcuda )
  target_include_directories(rtcuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_include_directories(rtcuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src)
  target_link_libraries(contra PRIVATE rtcuda)  
  
  install( TARGETS contra_cuda_rt rtcuda
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

  
  if (TARGET CUDA::cudart)
    target_link_libraries(contra PUBLIC CUDA::cudart CUDA::cuda_driver)
  else()
    target_link_libraries(contra PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
    target_include_directories(contra PUBLIC ${CUDA_INCLUDE_DIRS})
  endif()
  
  list(APPEND SUPPORTED_BACKENDS "cuda")
endif()

find_package(HIP)
if (HIP_FOUND)
  target_link_libraries(contra PUBLIC hip::host)
  target_compile_options(contra PRIVATE -D__HIP_HCC_COMPAT_MODE__=1)
  #set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_DIR}/cmake")  
  list(APPEND SUPPORTED_BACKENDS "rocm")

  find_program(ROCM_LD_LLD_EXE 
    NAMES 
      ld.lld  
    HINTS
      $ENV{ROCM_PATH}/llvm/bin
      $ENV{HIP_PATH}/../llvm/bin
      ENV ROCM_CLANG_PATH
      ENV HIP_CLANG_PATH
      ENV ROCM_LLVM_PATH
      ENV HIP_LLVM_PATH
    PATH_SUFFIXES
        bin
  )
  if (NOT ROCM_LD_LLD_EXE)
    MESSAGE(FATAL_ERROR "Could not find ld.lld.  Please set ROCM_LLVM_PATH")
  else ()
    MESSAGE(STATUS "Found ld.lld: ${ROCM_LD_LLD_EXE}")
  endif()
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
