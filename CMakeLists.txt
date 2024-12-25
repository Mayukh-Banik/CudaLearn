cmake_minimum_required(VERSION 3.31.3)
project(cuda_project LANGUAGES CUDA)

add_executable(cuda_app main.cu)

set(CMAKE_CUDA_ARCHITECTURES 86)

set_target_properties(cuda_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)
