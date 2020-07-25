
if (HIP_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm_utils.cpp)
endif()
