#pragma once

#include "defs/DoubleTensor.h"
#include <tuple>

DoubleTensor* empty(double val, std::string Device = "cpu");

DoubleTensor* empty(std::tuple<uint64_t> shape, std::string Device = "cpu");