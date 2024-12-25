# # bindings/CMakeLists.txt

# # Define the source file for bindings
# set(BINDINGS_SOURCES my_ext.cpp)

# # Create the shared library for bindings using nanobind (must come before setting properties)
# add_library(bindings MODULE ${BINDINGS_SOURCES})

# # Add nanobind include directory
# target_include_directories(bindings PRIVATE ${CMAKE_SOURCE_DIR}/ext/nanobind/include)
# target_include_directories(bindings PRIVATE ${CMAKE_SOURCE_DIR}/include)

# # Optionally, link nanobind if needed
# target_link_libraries(bindings PRIVATE nanobind)

# # Set target properties and ensure separable compilation is enabled for CUDA if needed
# set_target_properties(bindings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# bindings/CMakeLists.txt

# # Define the source file for bindings
# set(BINDINGS_SOURCES my_ext.cpp)

# # Use nanobind_add_module instead of add_library
# nanobind_add_module(my_ext ${BINDINGS_SOURCES})

# # Add include directories
# target_include_directories(my_ext PRIVATE ${CMAKE_SOURCE_DIR}/include)

# # Set CUDA properties if needed
# set_target_properties(my_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(BINDINGS_SOURCES my_ext.cpp)

# Use nanobind_add_module
nanobind_add_module(my_ext ${BINDINGS_SOURCES})

# Add include directories
target_include_directories(my_ext PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against your main cuda_project library
target_link_libraries(my_ext PRIVATE cuda_project)

# Set CUDA properties
set_target_properties(my_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)