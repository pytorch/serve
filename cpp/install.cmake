execute_process(COMMAND python -c "import ts; from pathlib import Path; print(Path(ts.__file__).parent / 'cpp')"
                OUTPUT_VARIABLE TARGET_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

message("Installing cpp backend into ${TARGET_DIR}")

if(EXISTS ${TARGET_DIR})
    execute_process(COMMAND rm -rf ${TARGET_DIR})
endif()

execute_process(COMMAND mkdir ${TARGET_DIR})
execute_process(COMMAND cp -rp ${CMAKE_BINARY_DIR}/bin ${TARGET_DIR}/bin)
execute_process(COMMAND cp -rp ${CMAKE_BINARY_DIR}/libs ${TARGET_DIR}/lib)
execute_process(COMMAND cp -rp ${CMAKE_BINARY_DIR}/resources ${TARGET_DIR}/resources)
