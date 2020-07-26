
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/args.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/ast.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/compiler.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/codegen.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/contra.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/device_jit.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/flow.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/futures.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/jit.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/leafs.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/lexer.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/loops.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/reductions.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/serial.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/serial_rt.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/serializer.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/tasking.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/token.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/vizualizer.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp )

if (CUDA_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_jit.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rt.cpp )
  
  target_sources( contra_cuda_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rt.cu)
  target_sources( contra_cuda_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_reduce.cu)
endif()
if (Kokkos_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_rt.cpp )
endif()
if (Legion_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/legion.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/legion_rt.cpp )
endif()
if (HIP_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm_jit.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm_rt.cpp )

  add_custom_target(contra_rocm_devlibs
    DEPENDS rocm_reduce.bc)
  add_dependencies(contra contra_rocm_devlibs)

  #set_property(TARGET contra_rocm_devlibs APPEND PROPERTY OBJECT_DEPENDS rocm_reduce.bc)

  add_custom_command(OUTPUT rocm_reduce.bc
    COMMAND
      ${ROCM_CLANG_EXE}
      -cl-std=CL2.0
      -Xclang 
      -finclude-default-header
      -xcl
      -target amdgcn-amd-amdhsa
      -mcpu=gfx900
      -c ${CMAKE_CURRENT_SOURCE_DIR}/rocm_reduce.cl
      -emit-llvm
      -I ${CMAKE_CURRENT_BINARY_DIR}/..
      -I ${CMAKE_CURRENT_SOURCE_DIR}/..
      -o rocm_reduce.bc
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/rocm_reduce.cl
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR})
endif()
