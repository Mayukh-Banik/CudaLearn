#pragma once

#include "core/DoubleTensor.h"
#include <nanobind/ndarray.h>

nanobind::ndarray<double, nanobind::c_contig> toPythonBufferProtocol(const DoubleTensor* tensor, std::string device = "cuda");