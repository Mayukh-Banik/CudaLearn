#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

__device__ inline double POWER(double a, double b) { return pow(a, b); }

__device__ inline double ZERO(double a) { return 0.0; }

__device__ inline double ONES(double a) { return 1.0; }
