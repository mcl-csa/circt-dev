add_custom_target(CIRCTUnitTests)
set_target_properties(CIRCTUnitTests PROPERTIES FOLDER "CIRCT Tests")

function(add_circt_unittest test_dirname)
  add_unittest(CIRCTUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
