#include "defs/CreationRoutines.h"
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <cmath>
#include <sstream>
#include "defs/CustomCudaMacros.h"

__global__ void deviceFillAllValuesWithConstant(double *Data, uint64_t NumElem, double val);

DoubleTensor *empty(uint64_t val, std::string Device)
{
    std::tuple<uint64_t, uint64_t> values = {val, 1};
    return empty(values, Device);
}

DoubleTensor *empty(std::vector<uint64_t> shape, std::string Device)
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

DoubleTensor *empty(std::tuple<uint64_t, uint64_t> shape, std::string Device)
{
    std::vector<uint64_t> val = {std::get<0>(shape), std::get<1>(shape)};
    return new DoubleTensor(val, Device);
}

__global__ void eyeHelper(double *Data, const uint64_t (&input)[2], int64_t K)
{
    uint64_t index = threadIdx.x + blockDim.x * blockIdx.x;
    K = K + index;
    if (K >= 0 && K < input[1] && index < input[0])
    {
        Data[index * input[0] + K] = 1.0;
    }
}

DoubleTensor *eye(uint64_t N, uint64_t M, int64_t K, std::string Device)
{
    M = M == 0 ? N : M;
    std::vector<uint64_t> val(N, M);
    DoubleTensor *Tensor = new DoubleTensor(val, Device);
    if (Tensor->OnGPU)
    {
        ALLOCATE_BLOCKS_THREADS_SINGLE_TENSOR_ALL_VALUES(Tensor)
        eyeHelper<<<Blocks, Threads>>>(Tensor->Data, {NRows, NColumns}, K);
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

DoubleTensor *identity(uint64_t N, std::string Device)
{
    return eye(N, 0, 0, Device);
}

DoubleTensor *zeros(uint64_t val, std::string Device)
{
    std::tuple<uint64_t, uint64_t> values = {val, 1};
    return zeros(values, Device);
}

DoubleTensor *zeros(std::vector<uint64_t> shape, std::string Device)
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
        return zeros(shape[0], Device);
    case 2:
        return zeros(std::make_tuple(shape[0], shape[1]), Device);
    default:
        throw std::invalid_argument("List is too long.");
    }
    return nullptr;
}

DoubleTensor *zeros(std::tuple<uint64_t, uint64_t> shape, std::string Device)
{
    std::vector<uint64_t> val = {std::get<0>(shape), std::get<1>(shape)};
    DoubleTensor *Tensor = new DoubleTensor(val, Device);
    if (Tensor->OnGPU)
    {
        cudaError_t err =
            CUDA_THROW_RUNTIME_ERROR_CHECK(err, cudaMemset(Tensor->Data, 0, Tensor->ElementCount * Tensor->ItemSize);)
    }
    else
    {
        memset(Tensor->Data, 0, Tensor->ElementCount * Tensor->ItemSize);
    }
    return Tensor;
}

DoubleTensor *ones(uint64_t val, std::string Device)
{
    std::tuple<uint64_t, uint64_t> values = {val, 1};
    return ones(values, Device);
}

DoubleTensor *ones(std::vector<uint64_t> shape, std::string Device)
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
        return ones(shape[0], Device);
    case 2:
        return ones(std::make_tuple(shape[0], shape[1]), Device);
    default:
        throw std::invalid_argument("List is too long.");
    }
    return nullptr;
}

DoubleTensor *ones(std::tuple<uint64_t, uint64_t> shape, std::string Device)
{
    std::vector<uint64_t> val = {std::get<0>(shape), std::get<1>(shape)};
    DoubleTensor *Tensor = new DoubleTensor(val, Device);
    if (Tensor->OnGPU)
    {
        ALLOCATE_BLOCKS_THREADS_SINGLE_TENSOR_ALL_VALUES(Tensor);
        deviceFillAllValuesWithConstant<<<Blocks, Threads>>>(Tensor->Data, Tensor->ElementCount, 1.0);
    }
    else
    {
        for (uint64_t i = 0; i < Tensor->ElementCount; i++)
        {
            Tensor->Data[i] = 1.0;
        }
    }
    return Tensor;
}

__global__ void deviceFillAllValuesWithConstant(double *Data, uint64_t NumElem, double val)
{
    uint64_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < NumElem)
    {
        Data[index] = val;
    }
}

DoubleTensor *fill(uint64_t val, double vals, std::string Device)
{
    std::tuple<uint64_t, uint64_t> values = {val, 1};
    return fill(values, vals, Device);
}

DoubleTensor *fill(std::vector<uint64_t> shape, double vals, std::string Device)
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
        return fill(shape[0], vals, Device);
    case 2:
        return fill(std::make_tuple(shape[0], shape[1]), vals, Device);
    default:
        throw std::invalid_argument("List is too long.");
    }
    return nullptr;
}

DoubleTensor *fill(std::tuple<uint64_t, uint64_t> shape, const double val, std::string Device)
{
    std::vector<uint64_t> vals = {std::get<0>(shape), std::get<1>(shape)};
    DoubleTensor *Tensor = new DoubleTensor(vals, Device);
    if (Tensor->OnGPU)
    {
        ALLOCATE_BLOCKS_THREADS_SINGLE_TENSOR_ALL_VALUES(Tensor)
        deviceFillAllValuesWithConstant<<<Blocks, Threads>>>(Tensor->Data, Tensor->ElementCount, val);
    }
    else
    {
        for (uint64_t i = 0; i < Tensor->ElementCount; i++)
        {
            Tensor->Data[i] = val;
        }
    }
    return Tensor;
}

DoubleTensor *array(double val, std::string Device)
{
    return new DoubleTensor(val, Device);
}

DoubleTensor *array(std::vector<std::vector<double>> values, std::string Device)
{
    DoubleTensor *Tensor = nullptr;

    Tensor = new DoubleTensor(values, Device);

    return Tensor;
}

DoubleTensor *array(nanobind::ndarray<double, nanobind::shape<-1, -1>, nanobind::any_contig> array, bool copy)
{
    std::ostringstream S;
    if (array.DeviceType == nanobind::device::cpu::value)
    {
        S << "cpu";
    }
    else
    {
        S << "cuda:" << array.device_id();
    }
    DoubleTensor *tensor = new DoubleTensor(S.str());
    tensor->ElementCount = array.size();
    tensor->Shape[0] = array.shape_ptr()[0];
    tensor->Shape[1] = array.shape_ptr()[1];
    tensor->Strides[0] = array.stride_ptr()[0];
    tensor->Strides[1] = array.stride_ptr()[1];
    tensor->Order[0] = array.Order;
    if (copy)
    {
        if (tensor->OnGPU)
        {
            cudaError_t err;
            CUDA_THROW_RUNTIME_ERROR_CHECK(err, cudaMemcpy(tensor->Data, array.data(), tensor->ElementCount * tensor->ItemSize, cudaMemcpyDeviceToDevice))
        }
        else
        {
            memcpy(tensor->Data, array.data(), tensor->ElementCount * tensor->ItemSize);
        }
    }
    else
    {
        tensor->Data = array.data();
    }
    return tensor;
}

DoubleTensor *array(nanobind::ndarray<double, nanobind::shape<-1>, nanobind::any_contig> array, bool copy)
{
    std::ostringstream S;
    if (array.DeviceType == nanobind::device::cpu::value)
    {
        S << "cpu";
    }
    else
    {
        S << "cuda:" << array.device_id();
    }
    DoubleTensor *tensor = new DoubleTensor(S.str());
    tensor->ElementCount = array.size();
    tensor->Shape[0] = array.shape_ptr()[0];
    tensor->Shape[1] = 1;
    tensor->Strides[0] = array.stride_ptr()[0];
    tensor->Strides[1] = 1;
    tensor->Order[0] = array.Order;
    if (copy)
    {
        if (tensor->OnGPU)
        {
            cudaError_t err = cudaMemcpy(tensor->Data, array.data(), tensor->ElementCount * tensor->ItemSize, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error copying over data in GPU memory, try setting copy to false.");
            }
        }
        else
        {
            memcpy(tensor->Data, array.data(), tensor->ElementCount * tensor->ItemSize);
        }
    }
    else
    {
        tensor->Data = array.data();
    }
    return tensor;
}