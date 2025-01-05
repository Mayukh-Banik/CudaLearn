#include "nanobind/nanobind.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include "defs/CreationRoutines.h"
#include <iostream>

namespace nb = nanobind;

DoubleTensor *Empty(const nb::object &obj, const std::string &device = "cpu")
{
    if (nb::isinstance<nb::int_>(obj))
    {
        uint64_t val = nb::cast<uint64_t>(obj);
        return empty(val, device);
    }
    else if (nb::isinstance<nb::list>(obj))
    {
        nb::list py_list = nb::cast<nb::list>(obj);
        std::vector<uint64_t> shape;
        for (size_t i = 0; i < py_list.size(); ++i)
        {
            shape.push_back(nb::cast<uint64_t>(py_list[i]));
        }
        return empty(shape, device);
    }
    else if (nb::isinstance<nb::tuple>(obj))
    {
        nb::tuple py_tuple = nb::cast<nb::tuple>(obj);
        if (py_tuple.size() == 2)
        {
            uint64_t val1 = nb::cast<uint64_t>(py_tuple[0]);
            uint64_t val2 = nb::cast<uint64_t>(py_tuple[1]);
            return empty(std::make_tuple(val1, val2), device);
        }
        else
        {
            throw std::invalid_argument("Expected a tuple of two uint64_t values.");
        }
    }
    else
    {
        throw std::invalid_argument("Expected int, list, or tuple.");
    }
}

void empty(nb::module_ &m)
{
    m.def("empty", &Empty, "Create a new DoubleTensor from various Python objects",
          nb::arg("obj"), nb::arg("device") = "cpu");
}