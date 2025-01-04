cmake_minimum_required(VERSION 3.31.1)

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(DEV_MODULE Development.Module)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# Options to enable bindings and tests
option(ENABLE_BINDINGS "Enable building bindings" ON)
option(ENABLE_TESTS "Enable building tests" OFF)

option(DEBUG "Enable debugging" OFF)

# Include and src are always compiled
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src PROPERTY COMPILE_DEFINITIONS DEBUG=$<BOOL:${DEBUG}>)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/googletest)

# Conditionally compile bindings (shared library)
if(ENABLE_BINDINGS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bindings)
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings PROPERTY COMPILE_DEFINITIONS DEBUG=$<BOOL:${DEBUG}>)
endif()

# Conditionally compile tests (executable)
if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Bindings Enabled: ${ENABLE_BINDINGS}")
message(STATUS "Tests Enabled: ${ENABLE_TESTS}")
