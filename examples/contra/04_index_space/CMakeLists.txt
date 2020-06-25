foreach(_backend ${SUPPORTED_BACKENDS})
  foreach(_test index_space)
    add_test(
      NAME example_${_test}_${_backend}
      COMMAND $<TARGET_FILE:contra> -b ${_backend} ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cta)
  endforeach()
endforeach()
