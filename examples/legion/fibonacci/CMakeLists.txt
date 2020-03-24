add_executable(legion_fibonacci fibonacci.cpp)
target_include_directories(legion_fibonacci PRIVATE ${Legion_INCLUDE_DIRS})
target_link_libraries(legion_fibonacci PRIVATE  ${Legion_LIBRARIES})
