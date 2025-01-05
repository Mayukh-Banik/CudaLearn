#include "nanobind/nanobind.h"

#include "defs/DoubleTensor.h"

namespace nb = nanobind;

extern void empty(nb::module_ &m);

void classDef(nb::module_ &m)
{
    nb::class_<DoubleTensor>(m, "DoubleTensor")
        .def("__str__", [](const DoubleTensor &self) -> std::string
                { return const_cast<DoubleTensor &>(self).toString(false); })
        .def("__repr__", [](const DoubleTensor &self) -> std::string
                { return const_cast<DoubleTensor &>(self).toString(true); });
}

NB_MODULE(_DoubleTensor, m)
{
    classDef(m);
    empty(m);
}