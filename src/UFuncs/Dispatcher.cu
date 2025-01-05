#include "ufuncs/Ufuncs.h"
#include "ufuncs/Functions.h"

void HostSerializedApply2Var(const double *a, const double *b, double *c, const uint64_t NUM, const double (*func)(double, double))
{
    for (uint64_t i = 0; i < NUM; i++)
    {
        c[i] = func(a[i], b[i]);
    }
}

void HostSerializedApply1Var(double *a, const uint64_t NUM, const double (*func)(double))
{
    for (uint64_t i = 0; i < NUM; i++)
    {
        a[i] = func(a[i]);
    }
}

__global__ void cudaApply(double* a, const double (*func)(double))
{
    uint64_t x = threadIdx.x + blockDim.x * blockIdx.x;
    a[x] = func(a[x]);
}

void sqrt(DoubleTensor* a)
{
    if (a->OnGPU)
    {
        cudaApply<<<1, 1>>>(a->Data, SQRT);
    }
    else
    {
        HostSerializedApply1Var(a->Data, a->ElementCount, SQRT);
    }
}