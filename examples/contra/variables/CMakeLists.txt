foreach(_test variables arrays1 arrays2 arrays3)
  create_test(
    NAME ${_test}
    COMMAND $<TARGET_FILE:contra> ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cta
    COMPARE stdout
    STANDARD ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.std)
endforeach()
