set(HAVE_LEGION ${Legion_FOUND})
set(HAVE_KOKKOS ${Kokkos_FOUND})
set(HAVE_CUDA ${CUDA_FOUND})

if (CUDA_FOUND)
  set(_path "$<TARGET_FILE:contra_cuda_rt>")
  file (GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/config.defs" 
    CONTENT "
#define CONTRA_CUDA_LIBRARY \"${_path}\"\n
#define CONTRA_CUDA_LIBRARY_INSTALLED \"${CMAKE_INSTALL_PREFIX}/lib/${_path}\"\n
"
  )
endif()


configure_file(config.hpp.in config.hpp @ONLY)

add_subdirectory(utils)
add_subdirectory(librt)
add_subdirectory(librtcuda)
add_subdirectory(contra)
