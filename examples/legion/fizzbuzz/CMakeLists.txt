add_executable(legion_fizzbuzz fizzbuzz.cpp)
target_include_directories(legion_fizzbuzz PRIVATE ${Legion_INCLUDE_DIRS})
target_link_libraries(legion_fizzbuzz PRIVATE  ${Legion_LIBRARIES})
