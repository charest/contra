add_executable(legion_index_space index_space.cpp)
target_include_directories(legion_index_space PRIVATE ${Legion_INCLUDE_DIRS})
target_link_libraries(legion_index_space PRIVATE  ${Legion_LIBRARIES})
