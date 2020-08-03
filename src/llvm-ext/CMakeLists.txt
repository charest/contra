if (HIP_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/Target/AMDGPU/AMDGPULowerAlloca.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/Target/AMDGPU/AMDGPUPrintfRuntimeBinding.cpp )
endif()
