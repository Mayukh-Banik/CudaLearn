#include "nanobind/nanobind.h"
#include "core/DoubleTensor.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void DoubleTensorBindings(nb::module_ &m)
{
    nb::class_<DoubleTensor>(m, "DoubleTensor")
        .def(nb::init<double, int, int>(), nb::arg("value"), nb::arg("device") = 0, nb::arg("readonly") = 0)
        .def("set_index", &DoubleTensor::setIndex, nb::arg("val"), nb::arg("index"))
        .def("get_index", &DoubleTensor::getIndex, nb::arg("index"))
        .def("__getitem__", [](DoubleTensor &self, uint64_t index) {
            return self.getIndex(index);
        })
        .def("__setitem__", [](DoubleTensor &self, uint64_t index, double value) {
            self.setIndex(value, index);
        })
        .def("shape", [](const DoubleTensor &self) {
            return std::vector<uint64_t>(self.shape, self.shape + self.ndim);
        })
        .def("strides", [](const DoubleTensor &self) {
            return std::vector<uint64_t>(self.strides, self.strides + self.ndim);
        });
}
