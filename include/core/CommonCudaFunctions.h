#pragma once

#include "core/CommonCudaMacros.h"
#include <cstdint>

/**
 * Applied function pointer to all elements from double* -> uint64_t
 */
__global__ void applyFunctionToAllElements(double*, const uint64_t, double (*)(double));

/**
 * If apply returns true, then applies function to that index in array
 */
__global__ void applyFunctionToSomeElements(double *src, const uint64_t elementCount,
                                            double (*func)(double), bool (*apply)(uint64_t));

__global__ void applyValueToAllElements(double*, const uint64_t, const double);