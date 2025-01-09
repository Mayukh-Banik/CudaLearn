#pragma once

#include "core/DoubleTensor.h"
#include "nanobind/ndarray.h"
#include <tuple>

DoubleTensor* empty(const nanobind::ndarray<double, nanobind::c_contig>& array);
DoubleTensor* empty(const uint64_t val);
DoubleTensor* empty(const std::vector<uint64_t>& vals);