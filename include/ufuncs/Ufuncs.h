#pragma once

#include "defs/DoubleTensor.h"

void HostSerializedApply2Var(const double *a, const double *b, double *c, const uint64_t NUM, const double (*func)(double, double));

void HostSerializedApply1Var(double *a, const uint64_t NUM, const double (*func)(double));
