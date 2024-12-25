#include "temp.h"
#include "nanobind/nanobind.h"

NB_MODULE(my_ext, m) {
    m.def("add", &add);
}