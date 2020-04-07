add_executable(legion_hello hello.cpp)
target_include_directories(legion_hello PRIVATE ${Legion_INCLUDE_DIRS})
target_link_libraries(legion_hello PRIVATE  ${Legion_LIBRARIES})
