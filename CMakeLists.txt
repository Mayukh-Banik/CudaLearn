cmake_minimum_required(VERSION 3.31.3)

project(cuda_project LANGUAGES CXX CUDA)

set(DEV_MODULE Development.Module)

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# Options to enable bindings and tests
option(ENABLE_BINDINGS "Enable building bindings" OFF)
option(ENABLE_TESTS "Enable building tests" OFF)

# Include and src are always compiled
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

# Conditionally compile bindings (shared library)
if(ENABLE_BINDINGS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bindings)
endif()

# Conditionally compile tests (executable)
if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Bindings Enabled: ${ENABLE_BINDINGS}")
message(STATUS "Tests Enabled: ${ENABLE_TESTS}")
