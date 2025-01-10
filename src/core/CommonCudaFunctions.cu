#include "core/CommonCudaFunctions.h"
#include "core/CommonCudaMacros.h"

__global__ void applyFunctionToAllElements(double *src, const uint64_t elementCount, double (*func)(double))
{
    uint64_t index = GLOBAL_THREAD_INDEX_X;
    if (index < elementCount)
    {
        src[index] = func(src[index]);
    }
}

__global__ void applyValueToAllElements(double* src, const uint64_t elementCount, const double val)
{
    uint64_t index = GLOBAL_THREAD_INDEX_X;
    if (index < elementCount)
    {
        src[index] = val;
    }
}