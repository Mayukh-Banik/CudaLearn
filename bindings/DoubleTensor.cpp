#include "nanobind/nanobind.h"

template <typename T>
T add(T a, T b)
{
    return a + b;
}

template <typename T>
T sub(T a, T b)
{
    return a - b;
}

NB_MODULE(_DoubleTensor, m) {
    m.def("add", &add<int>);
    m.def("sub", &sub<int>);
}