#include "core/TypeConversions.h"
#include "core/CommonCudaMacros.h"

nanobind::ndarray<double, nanobind::c_contig> toPythonBufferProtocol(const DoubleTensor *tensor, std::string device)
{
    if (tensor == NULL || tensor == nullptr)
    {
        throw std::runtime_error("Pointer passed was invalid.");
    }
    if (device == "cpu")
    {
        double *data = new double[tensor->elementCount];
        cudaError_t err;
        err = cudaMemcpy(data, tensor->buf, tensor->len, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            delete[] data;
            throw std::runtime_error(cudaGetErrorString(err));
        }
        nanobind::capsule owner(data, [](void *p) noexcept 
                                { delete[] (double *)p; });
        return nanobind::ndarray<double, nanobind::c_contig>(
            data,
            tensor->ndim,
            tensor->shape,
            owner,
            tensor->strides
        );
    }
    else
    {
        throw std::runtime_error("Unsupported device type: " + device);
    }
}