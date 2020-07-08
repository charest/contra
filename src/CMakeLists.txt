set(HAVE_LEGION ${Legion_FOUND})
set(HAVE_KOKKOS ${Kokkos_FOUND})
set(HAVE_CUDA ${CUDA_FOUND})

if (CUDA_FOUND)
  file (GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/config.defs" 
    CONTENT "
#define CONTRA_CUDA_LIBRARY \"$<TARGET_FILE:contra_cuda>\"\n
#define CONTRA_CUDA_LIBRARY_INSTALLED \"${CMAKE_INSTALL_PREFIX}/lib/$<TARGET_FILE_NAME:contra_cuda>\"\n
"
  )
endif()


configure_file(config.hpp.in config.hpp @ONLY)

add_subdirectory(utils)
add_subdirectory(librt)
add_subdirectory(contra)
