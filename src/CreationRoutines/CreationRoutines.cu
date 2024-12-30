#include "defs/CreationRoutines.h"
#include <stdexcept>
#include <cuda_runtime_api.h>

DoubleTensor *empty(uint64_t val, std::string Device = "cpu")
{
    std::tuple<uint64_t, uint64_t> values = {val, 1};
    return empty(values, Device);
}

DoubleTensor *empty(std::vector<uint64_t> shape, std::string Device = "cpu")
{
    if (shape.empty())
    {
        throw std::invalid_argument("How did you get here, list is empty.");
    }
    switch (shape.size())
    {
    case 0:
        throw std::invalid_argument("How did you get here, list is empty.");
        break;
    case 1:
        return empty(shape[0], Device);
    case 2:
        return empty(std::make_tuple(shape[0], shape[1]), Device);
    default:
        throw std::invalid_argument("List is too long.");
    }
    return nullptr;
}

DoubleTensor *empty(std::tuple<uint64_t, uint64_t> shape, std::string Device = "cpu")
{
    std::vector<uint64_t> val = {std::get<0>(shape), std::get<1>(shape)};
    return new DoubleTensor(val, Device);
}

DoubleTensor *eye(uint64_t N, uint64_t M = 0, int64_t K = 0, std::string Device = "cpu")
{
    M = M == 0 ? N : M;
    std::vector<uint64_t> val(N, M);
    DoubleTensor *Tensor = new DoubleTensor(val, Device);
    if (Tensor->OnGPU)
    {
    }
    else
    {
        for (uint64_t i = 0; i < Tensor->Shape[0]; i++, K++)
        {
            if (K < 0)
            {
                continue;
            }
            else if (K >= Tensor->Shape[1])
            {
                break;
            }
            Tensor->Data[(i * Tensor->Shape[1]) + K] = 1.0;
        }
    }
    return Tensor;
}