
if (CUDA_FOUND)
  target_sources( rtcuda PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_utils.cu)
endif()
