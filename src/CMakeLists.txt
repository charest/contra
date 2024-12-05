set(HAVE_LEGION ${Legion_FOUND})
set(HAVE_CUDA ${CUDA_FOUND})
set(HAVE_ROCM ${HIP_FOUND})
set(HAVE_THREADS ${Threads_FOUND})
set(HAVE_MPI ${MPI_FOUND})

if (CUDA_FOUND)
  set(_path "$<TARGET_FILE:contra_cuda_rt>")
  file (GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cuda.defs" 
    CONTENT "
#define CONTRA_CUDA_LIBRARY \"${_path}\"\n
#define CONTRA_CUDA_LIBRARY_INSTALLED \"${CMAKE_INSTALL_PREFIX}/lib/${_path}\"\n
"
  )
endif()

if (HIP_FOUND)
  foreach(_tgt ${AMD_DEVICE_LIBS_TARGETS})
    set(_path "$<TARGET_FILE:${_tgt}>")
    string(TOUPPER ${_tgt} _def)
    set(_content "${_content}\n#define ${_def}_PATH \"${_path}\"")
  endforeach()
  file (GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/rocm.defs" 
    CONTENT "${_content}")
endif()

configure_file(config.hpp.in config.hpp @ONLY)
configure_file(config.h.in config.h @ONLY)

add_subdirectory(utils)
add_subdirectory(librt)
add_subdirectory(librtcuda)
add_subdirectory(librtrocm)
add_subdirectory(llvm-ext)
add_subdirectory(contra)
