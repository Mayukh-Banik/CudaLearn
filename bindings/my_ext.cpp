#include "nanobind/nanobind.h"

template <typename T>
T add(T a, T b)
{
    return a + b;
}

NB_MODULE(my_ext, m) {
    m.def("add", &add<int>);
}