add_executable(legion_index_task index_task.cpp)
target_include_directories(legion_index_task PRIVATE ${Legion_INCLUDE_DIRS})
target_link_libraries(legion_index_task PRIVATE  ${Legion_LIBRARIES})
