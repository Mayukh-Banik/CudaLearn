#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

double add(double, double);
double sub(double, double);
double mul(double, double);
double div(double, double);

__device__ __host__ inline const double SQRT(double);

double sin(double);
double cos(double);
double tan(double);
double sec(double);
double csc(double);
double cot(double);

__device__ __host__ inline void var(uint64_t);