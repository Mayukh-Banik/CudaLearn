#pragma once

#include "defs/DoubleTensor.h"
#include <tuple>
#include "nanobind/ndarray.h"

/**
 * @brief Numpy empty method, default to 1, 1 matrixes
 */
DoubleTensor *empty(uint64_t val, std::string Device = "cpu");
DoubleTensor *empty(std::vector<uint64_t> shape, std::string Device = "cpu");
DoubleTensor *empty(std::tuple<uint64_t, uint64_t> shape, std::string Device = "cpu");

DoubleTensor *eye(uint64_t N, uint64_t M = 0, int64_t K = 0, std::string Device = "cpu");

DoubleTensor *identity(uint64_t N, std::string Device = "cpu");

DoubleTensor *zeros(uint64_t val, std::string Device = "cpu");
DoubleTensor *zeros(std::vector<uint64_t> shape, std::string Device = "cpu");
DoubleTensor *zeros(std::tuple<uint64_t, uint64_t> shape, std::string Device = "cpu");

DoubleTensor *ones(uint64_t val, std::string Device = "cpu");
DoubleTensor *ones(std::vector<uint64_t> shape, std::string Device = "cpu");
DoubleTensor *ones(std::tuple<uint64_t, uint64_t> shape, std::string Device = "cpu");

DoubleTensor *fill(uint64_t val, double fill, std::string Device = "cpu");
DoubleTensor *fill(std::vector<uint64_t> shape, double fill, std::string Device = "cpu");
DoubleTensor *fill(std::tuple<uint64_t, uint64_t> shape, double fill, std::string Device = "cpu");

DoubleTensor *array(double val, std::string Device = "cpu");
DoubleTensor *array(std::vector<std::vector<double>> values, std::string Device = "cpu");
DoubleTensor *array(nanobind::ndarray<double, nanobind::shape<-1, -1>, nanobind::any_contig> array, bool copy = false);
DoubleTensor *array(nanobind::ndarray<double, nanobind::shape<-1>, nanobind::any_contig> array, bool copy = false);