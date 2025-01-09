#pragma once

#include <cuda_runtime_api.h>

#define CUDA_CHECK_ERROR(err, function, throwType) \
    err = function;                                \
    if (err != cudaSuccess)                        \
    {                                              \
        throw throwType(cudaGetErrorString(err));  \
    }
