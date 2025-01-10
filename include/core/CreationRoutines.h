#pragma once

#include "core/DoubleTensor.h"
#include "nanobind/ndarray.h"
#include <tuple>

DoubleTensor* empty(const uint64_t val);
DoubleTensor* empty(const std::vector<uint64_t>& vals);

DoubleTensor* eye(const uint64_t N, const uint64_t M = 0, const int64_t k = 0);
DoubleTensor* identity(const uint64_t N);
DoubleTensor* ones(const uint64_t val);
DoubleTensor* ones(const std::vector<uint64_t>& vals);
DoubleTensor* zeros(const uint64_t val);
DoubleTensor* zeros(const std::vector<uint64_t>& vals);
DoubleTensor* full(const uint64_t val, const double FILL);
DoubleTensor* full(const std::vector<uint64_t>& vals, const double FILL);
