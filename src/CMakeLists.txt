# List of source files to compile
file(GLOB_RECURSE SOURCE_FILES
    "*.cu"
    "*.cpp"
    "*.c"
)

# Compile all sources in src/ directory
add_library(cuda_project ${SOURCE_FILES})

# Link to include directory for headers
target_include_directories(cuda_project PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(cuda_project PROPERTIES LINKER_LANGUAGE CUDA)

# Enable CUDA if needed
set_target_properties(cuda_project PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
