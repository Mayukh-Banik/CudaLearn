#pragma once

#include "cuda_runtime_api.h"
#include <cmath>

#define ALLOCATE_BLOCKS_THREADS_SINGLE_TENSOR_ALL_VALUES(x)                                   \
    const uint64_t NRows = x->Shape[0], NColumns = x->Shape[1];                               \
    const uint32_t MaxThreads = x->deviceProperties.MaxThreadsPerBlock;                       \
    const uint32_t WarpSize = x->deviceProperties.WarpSize;                                   \
    dim3 Blocks(                                                                              \
        std::ceil((double)NRows / (double)MaxThreads),                                        \
        1, 1);                                                                                \
    dim3 Threads(                                                                             \
        NRows >= MaxThreads - WarpSize ? MaxThreads : std::ceil(NRows / WarpSize) * WarpSize, \
        1, 1);

#define CUDA_THROW_RUNTIME_ERROR_CHECK(err, x)             \
    err = x;                                               \
    if (err != cudaSuccess)                                \
    {                                                      \
        throw std::runtime_error(cudaGetErrorString(err)); \
    }