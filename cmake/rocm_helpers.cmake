############################################################################$###
# Write bitcode
############################################################################$###
macro(compile_to_bitcode target)

  set(options)
  set(oneValueArgs SOURCE TARGET OUTPUT)
  set(multiValueArgs DEPENDS INCLUDES)
  cmake_parse_arguments(
    ARGS
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN} )

  foreach(_gpu ${AMDGPU_TARGETS})

    set(_target ${target}_${_gpu})
    
    string(REGEX REPLACE "\\.[^.]*$" "" _output_no_ext ${ARGS_OUTPUT})
    get_filename_component(_output_ext ${ARGS_OUTPUT} LAST_EXT)
    set(_output ${_output_no_ext}.${_gpu}${_output_ext})
    
    MESSAGE(STATUS "Compiling ${_target} to bitcode")
    add_custom_target(${_target} DEPENDS ${_output})

    set(_target_includes)
    if (ARGS_INCLUDES)
      foreach(_include ${ARGS_INCLUDES})
        list(APPEND _target_includes -I${_include})
      endforeach()
    endif()

    get_filename_component(_extension ${ARGS_SOURCE} LAST_EXT)
    if (_extension STREQUAL ".ll")
      
      add_custom_command(OUTPUT ${_output}
        COMMAND
          ${ROCM_CLANG_EXE}
          -Xclang 
          -xasm
          -target amdgcn-amd-amdhsa
          -mcpu=${_gpu}
          -c ${ARGS_SOURCE}
          -emit-llvm
          ${_target_includes}
          -o ${_output}
        DEPENDS
          ${ARGS_SOURCE}
          ${ARGS_DEPENDS}
        WORKING_DIRECTORY
          ${CMAKE_CURRENT_BINARY_DIR})

    elseif(_extension STREQUAL ".cl")

      add_custom_command(OUTPUT ${_output}
        COMMAND
          ${ROCM_CLANG_EXE}
          -cl-std=CL2.0
          -Xclang 
          -finclude-default-header
          -xcl
          -target amdgcn-amd-amdhsa
          -mcpu=${_gpu}
          -c ${ARGS_SOURCE}
          -emit-llvm
          ${_target_includes}
          -o ${_output}
        DEPENDS
          ${ARGS_SOURCE}
          ${ARGS_DEPENDS}
        WORKING_DIRECTORY
          ${CMAKE_CURRENT_BINARY_DIR})

    endif()

    if (ARGS_TARGET)
      add_dependencies(${ARGS_TARGET} ${_target})
    endif()

  endforeach() # GPU TARGETS

endmacro()

############################################################################$###
# Find an executable
############################################################################$###
macro (find_rocm_program var)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs NAMES)
  cmake_parse_arguments(
    ARGS
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN} )

  find_program(${var}
    NAMES 
      ${ARGS_NAMES}
    HINTS
      $ENV{ROCM_PATH}/llvm/bin
      $ENV{HIP_PATH}/../llvm/bin
      ENV ROCM_CLANG_PATH
      ENV HIP_CLANG_PATH
      ENV ROCM_LLVM_PATH
      ENV HIP_LLVM_PATH
    PATH_SUFFIXES
        bin
  )
  
  if (NOT ${var})
    MESSAGE(FATAL_ERROR "Could not find ${ARGS_NAMES}.  Please set ROCM_LLVM_PATH")
  else ()
    MESSAGE(STATUS "Found ${ARGS_NAMES}: ${${var}}")
  endif()

endmacro()
