# set(TEST_SOURCES main_test.cpp)
file(GLOB_RECURSE TEST_SOURCES
    "*.cpp"
    "*.cu"
)

# Create the test executable
add_executable(tests ${TEST_SOURCES})

# Include directories
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link the test executable against Google Test and the main project
target_link_libraries(tests PRIVATE
    gtest
    gtest_main
    cuda_project
)

set_target_properties(cuda_project PROPERTIES LINKER_LANGUAGE CUDA)

# Optionally, set CUDA separable compilation if needed
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add a CMake target for running the tests
add_test(NAME tests COMMAND tests)