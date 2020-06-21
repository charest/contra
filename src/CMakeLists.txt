set(HAVE_LEGION ${Legion_FOUND})
set(HAVE_KOKKOS ${Kokkos_FOUND})
configure_file(config.hpp.in config.hpp @ONLY)

add_subdirectory(utils)
add_subdirectory(librt)
add_subdirectory(contra)
