# List of source files to compile
find_package(Python3 COMPONENTS Development REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
    "*.cu"
    "*.cpp"
    "*.c"
)

# Compile all sources in src/ directory
add_library(cuda_project ${SOURCE_FILES})

# Link to include directory for headers
target_include_directories(cuda_project PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ext/nanobind/include
    ${Python3_INCLUDE_DIRS})

set_target_properties(cuda_project PROPERTIES LINKER_LANGUAGE CUDA)

set_target_properties(cuda_project PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

# Enable CUDA if needed
set_target_properties(cuda_project PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_compile_options(cuda_project PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)

target_compile_options(cuda_project PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        -Xcompiler=-Werror
        -Xcompiler=-fPIC
        --expt-relaxed-constexpr
        --extended-lambda
        --use_fast_math
        --ptxas-options=-v
        --maxrregcount=128
        -Xptxas=-O3
    >
)

target_compile_options(cuda_project PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
        -Werror
        -O3
        -fPIC
    >
)