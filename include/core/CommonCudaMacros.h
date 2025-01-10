#pragma once

#include <cuda_runtime_api.h>

#define CUDA_CHECK_ERROR(err, function, throwType) \
    err = function;                                \
    if (err != cudaSuccess)                        \
    {                                              \
        throw throwType(cudaGetErrorString(err));  \
    }

#define LAUNCH_BLOCKS(numElems, maxThreads) numElems > maxThreads ? ((numElems / maxThreads) + 1) : 1

#define LAUNCH_THREADS_LESS_THAN_MAX_THREADS(numElems, warpSize) (numElems / warpSize) + 1

#define GLOBAL_THREAD_INDEX_X threadIdx.x + blockDim.x * blockIdx.x
