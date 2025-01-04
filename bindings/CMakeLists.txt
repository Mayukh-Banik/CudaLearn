# set(BINDINGS_SOURCES my_ext.cpp)
file(GLOB_RECURSE BINDINGS_SOURCES
    "*.cpp"
)

# Use nanobind_add_module
nanobind_add_module(_DoubleTensor ${BINDINGS_SOURCES})

set(ROOT_DIR ${CMAKE_SOURCE_DIR})
set(DOUBLE_TENSOR_DIR ${ROOT_DIR}/DoubleTensor)

# Create output directory for shared library
file(MAKE_DIRECTORY ${DOUBLE_TENSOR_DIR})

# Add include directories
target_include_directories(_DoubleTensor PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against your main cuda_project library
target_link_libraries(_DoubleTensor PRIVATE cuda_project)

# Set CUDA properties
set_target_properties(_DoubleTensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(_DoubleTensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${DOUBLE_TENSOR_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${DOUBLE_TENSOR_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${DOUBLE_TENSOR_DIR}
)