
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/args.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/ast.cpp )
target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/communicator.cpp )
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

if (MPI_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/communicator_mpi.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_rt.cpp )
endif()

if (CUDA_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_jit.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rt.cpp )
  
  target_sources( contra_cuda_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rt.cu)
  target_sources( contra_cuda_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_reduce.cu)
endif()
if (Legion_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/legion.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/legion_rt.cpp )
endif()
if (HIP_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm_jit.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/rocm_rt.cpp )

  compile_to_bitcode(
    rocm_reduce_bc   
    SOURCE
      ${CMAKE_CURRENT_SOURCE_DIR}/rocm_reduce.cl
    OUTPUT
      rocm_reduce.bc
    INCLUDES
      ${CMAKE_CURRENT_BINARY_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/..
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/../config.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../librtrocm/rocm_builtins.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../librtrocm/rocm_scratch.h
    TARGET
      contra_rocm_devlibs
    )
endif()
if (Threads_FOUND)
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/threads.cpp )
  target_sources( contra PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/threads_rt.cpp )
endif()
