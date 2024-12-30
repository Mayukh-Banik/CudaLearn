#pragma once

#include "defs/DoubleTensor.h"
#include <tuple>

/**
 * @brief Numpy empty method, default to 1, 1 matrixes
 */
DoubleTensor *empty(uint64_t val, std::string Device = "cpu");
DoubleTensor *empty(std::vector<uint64_t> shape, std::string Device = "cpu");
DoubleTensor *empty(std::tuple<uint64_t, uint64_t> shape, std::string Device = "cpu");

DoubleTensor *eye(uint64_t N, uint64_t M = 0, int64_t K = 0, std::string Device = "cpu");

DoubleTensor *identity(uint64_t N, std::string Device = "cpu");
