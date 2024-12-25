set(TEST_SOURCES main_test.cpp)

add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tests PRIVATE cuda_project)  # Link against the main project if needed
