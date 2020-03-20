add_executable(hello-legion hello.cpp)
target_include_directories(hello-legion PRIVATE ${Legion_INCLUDE_DIRS})
target_link_libraries(hello-legion PRIVATE  ${Legion_LIBRARIES})
