foreach(_test hello comments print)
  create_test(
    NAME test_${_test}
    COMMAND $<TARGET_FILE:contra> --dump-dot ${_test}.dot ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cta
    COMPARE
      stdout
      ${_test}.dot
    STANDARD
      ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.std
      ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.dot.std)
endforeach()
