add_executable(hpx_hello hello.cpp)
target_link_libraries(hpx_hello PRIVATE HPX::hpx HPX::iostreams_component)
