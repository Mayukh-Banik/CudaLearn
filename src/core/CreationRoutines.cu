#include "core/CreationRoutines.h"

DoubleTensor *empty(const nanobind::ndarray<double, nanobind::c_contig> &array)
{
    DoubleTensor *tensor = new DoubleTensor(array.size());
    tensor->ndim = array.ndim();
    tensor->shape = (uint64_t *)malloc(tensor->ndim * sizeof(double));
    tensor->strides = (uint64_t *)malloc(tensor->ndim * sizeof(double));
    if (tensor->strides == NULL || tensor->shape == NULL)
    {
        throw std::runtime_error("Malloc failure in copy.");
    }
    for (uint64_t i = 0; i < tensor->ndim; i++)
    {
        tensor->shape[i] = array.shape_ptr()[i];
        tensor->strides[i] = array.stride_ptr()[i];
    }
    cudaError_t err = cudaMalloc((void **)&tensor->buf, tensor->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate memory for new array");
    }
    err = cudaMemcpy((void *)tensor->buf, (void *)array.data(),
                     tensor->len,
                     array.device_type() == nanobind::device::cpu::value
                         ? cudaMemcpyHostToDevice
                         : cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy over memory in same GPU");
    }
    return tensor;
}

DoubleTensor *empty(const uint64_t val)
{
    std::vector<uint64_t> values(val);
    return empty(values);
}

DoubleTensor *empty(const std::vector<uint64_t> &vals)
{
    DoubleTensor *tensor = new DoubleTensor(vals);
    return tensor;
}