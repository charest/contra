add_executable(kokkos_hello hello.cpp)
target_link_libraries(kokkos_hello PRIVATE Kokkos::kokkos)
