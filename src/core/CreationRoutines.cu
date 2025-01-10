#include "core/CreationRoutines.h"
#include "core/CommonCudaMacros.h"

#include "core/CommonCudaFunctions.h"
#include "funcs/Basic.h"
#include <iostream>

DoubleTensor *empty(const uint64_t val)
{
    std::vector<uint64_t> values = {val};
    return empty(values);
}

DoubleTensor *empty(const std::vector<uint64_t> &vals)
{
    DoubleTensor *tensor = new DoubleTensor(vals);
    return tensor;
}

__global__ void eyeHelper(double *src, const uint64_t N, const uint64_t M, const int64_t k)
{
    uint64_t index = GLOBAL_THREAD_INDEX_X;
    int64_t row = index;
    int64_t col = row + k;
    if (row < N && col >= 0 && col < M)
    {
        src[row * M + col] = 1.0;
    }
}

DoubleTensor *eye(const uint64_t N, const uint64_t M, const int64_t k)
{
    std::vector<uint64_t> vec = {N, M == 0 ? N : M};
    // vec.push_back(M == 0 ? N : M);
    DoubleTensor *tensor = zeros(vec);
    uint64_t blocks = N / tensor->cudaProperties.maxThreadsPerBlock + 1;
    uint64_t threads = tensor->cudaProperties.maxThreadsPerBlock;

    std::cerr << "Blocks: " << blocks << std::endl;
    std::cerr << "Threads: " << threads << std::endl;

    eyeHelper<<<blocks, threads>>>(tensor->buf, vec[0], vec[1], k);
    return tensor;
}

DoubleTensor *identity(const uint64_t N)
{
    return eye(N);
}

DoubleTensor *ones(const uint64_t val)
{
    std::vector<uint64_t> values = {val};
    return ones(values);
}

DoubleTensor *ones(const std::vector<uint64_t> &vals)
{
    return full(vals, 1);
}

DoubleTensor *zeros(const uint64_t val)
{
    std::vector<uint64_t> values = {val};
    return zeros(values);
}

DoubleTensor *zeros(const std::vector<uint64_t> &vals)
{
    return full(vals, 0);
}

DoubleTensor *full(const uint64_t val, const double FILL)
{
    std::vector<uint64_t> values = {val};
    return full(values, FILL);
}

DoubleTensor *full(const std::vector<uint64_t> &vals, const double FILL)
{
    DoubleTensor *tensor = new DoubleTensor(vals);
    uint64_t blocks = LAUNCH_BLOCKS(
        tensor->elementCount, tensor->cudaProperties.maxThreadsPerBlock);
    uint64_t numThreads =
        tensor->elementCount < tensor->cudaProperties.maxThreadsPerBlock
            ? tensor->cudaProperties.maxThreadsPerBlock
            : LAUNCH_THREADS_LESS_THAN_MAX_THREADS(
                  tensor->elementCount, tensor->cudaProperties.warpSize);
    applyValueToAllElements<<<blocks, numThreads>>>(tensor->buf, tensor->elementCount, FILL);
    return tensor;
}