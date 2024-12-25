# set(BINDINGS_SOURCES my_ext.cpp)
file(GLOB_RECURSE BINDINGS_SOURCES
    "*.cpp"
)

# Use nanobind_add_module
nanobind_add_module(my_ext ${BINDINGS_SOURCES})

# Add include directories
target_include_directories(my_ext PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against your main cuda_project library
target_link_libraries(my_ext PRIVATE cuda_project)

# Set CUDA properties
set_target_properties(my_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)