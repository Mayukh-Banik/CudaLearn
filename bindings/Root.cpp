#include "nanobind/nanobind.h"

namespace nb = nanobind;

void DoubleTensorBindings(nb::module_ &m);

NB_MODULE(_DoubleTensor, m)
{
    DoubleTensorBindings(m);
}