
if (HIP_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm_utils.cpp)
  
  compile_to_bitcode(
    rocm_scratch_bc
    SOURCE
      ${CMAKE_CURRENT_SOURCE_DIR}/rocm_scratch.ll
    OUTPUT
      rocm_scratch.bc
    TARGET
      contra_rocm_devlibs
    )
endif()
