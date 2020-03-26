foreach(_test fizz_buzz)
  add_test(
    NAME example_${_test}
    COMMAND $<TARGET_FILE:contra> ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cta)
endforeach()
