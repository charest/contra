add_executable(kokkos_index_task index_task.cpp)
target_link_libraries(kokkos_index_task PRIVATE Kokkos::kokkos)
