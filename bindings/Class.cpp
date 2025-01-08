#include "nanobind/nanobind.h"
#include "core/DoubleTensor.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <iostream>

namespace nb = nanobind;

void DoubleTensorBindings(nb::module_ &m)
{
     nb::class_<DoubleTensor>(m, "DoubleTensor")
         .def(nb::init<double>(), nb::arg("value"))
         .def(nb::init<std::vector<double>>(), nb::arg("values"))
         .def(nb::init<std::vector<std::vector<double>>>(), nb::arg("values"))
         .def("set_index", &DoubleTensor::setIndex, nb::arg("val"), nb::arg("index"))
         .def("get_index", &DoubleTensor::getIndex, nb::arg("index"))
         .def("__getitem__", [](DoubleTensor &self, uint64_t index)
              { return self.getIndex(index); })
         .def("__setitem__", [](DoubleTensor &self, uint64_t index, double value)
              { self.setIndex(value, index); })
         .def_prop_ro("shape", [](const DoubleTensor &self)
                      {  if (self.ndim == 0) { return nb::make_tuple(); }
                std::vector<uint64_t> shape_vec(self.shape, self.shape + self.ndim);
                return nb::make_tuple(shape_vec); })
         .def_prop_ro("strides", [](const DoubleTensor &self)
                      {  if (self.ndim == 0) {  return nb::make_tuple(); }
                std::vector<uint64_t> strides_vec(self.strides, self.strides + self.ndim);
                return nb::make_tuple(strides_vec); })
         .def_prop_ro("deviceNumber", [](const DoubleTensor &self)
                      { return self.deviceNumber; })
         .def_prop_ro("device", [](const DoubleTensor &self)
                      { return "cuda:" + std::to_string(self.deviceNumber); });
}
