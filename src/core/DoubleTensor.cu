#include "core/DoubleTensor.h"

#include "cuda_runtime_api.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <sstream>

void DoubleTensor::setDeviceProperties()
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceCount(&this->deviceNumber);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    if (this->deviceNumber == 0)
    {
        throw std::runtime_error("No CUDA GPUS found");
    }
    err = cudaGetDevice(&this->deviceNumber);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaGetDeviceProperties(&prop, this->deviceNumber);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    strncpy(this->cudaProperties.name, prop.name, sizeof(this->cudaProperties.name) - 1);
    this->cudaProperties.name[sizeof(this->cudaProperties.name) - 1] = '\0';
    this->cudaProperties.warpSize = prop.warpSize;
    this->cudaProperties.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    this->cudaProperties.maxThreadsDim[0] = prop.maxThreadsDim[0];
    this->cudaProperties.maxThreadsDim[1] = prop.maxThreadsDim[1];
    this->cudaProperties.maxThreadsDim[2] = prop.maxThreadsDim[2];
    this->cudaProperties.maxGridSize[0] = prop.maxGridSize[0];
    this->cudaProperties.maxGridSize[1] = prop.maxGridSize[1];
    this->cudaProperties.maxGridSize[2] = prop.maxGridSize[2];
    this->cudaProperties.maxGrids = prop.maxGridSize[0];
}

DoubleTensor::DoubleTensor(double val)
{
    this->len = this->itemsize;
    this->ndim = 0;
    this->shape = NULL;
    this->strides = NULL;
    this->elementCount = 1;
    setDeviceProperties();
    cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMemcpy(this->buf, &val, this->itemsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

DoubleTensor::DoubleTensor(std::vector<double> vector)
{
    this->len = vector.size() * this->itemsize;
    this->ndim = 1;
    this->shape = (uint64_t *)malloc(sizeof(*this->shape) * ndim);
    this->strides = (uint64_t *)malloc(sizeof(*this->shape) * ndim);
    if (this->strides == NULL || this->shape == NULL)
    {
        throw std::runtime_error("Malloc failure in copy.");
    }
    this->shape[0] = vector.size();
    this->strides[0] = this->itemsize;
    this->elementCount = vector.size();
    setDeviceProperties();
    cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMemcpy(this->buf, vector.data(), this->len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

DoubleTensor::DoubleTensor(std::vector<std::vector<double>> vector)
{
    if (vector.empty() || vector[0].empty())
    {
        throw std::runtime_error("Empty vector");
    }
    size_t row_size = vector[0].size();
    for (uint64_t i = 1; i < vector.size(); i++)
    {
        if (vector[i].size() != row_size)
        {
            throw std::runtime_error("Vector rows aren't all the same size");
        }
    }
    this->len = vector.size() * row_size * this->itemsize;
    this->ndim = 2;
    this->shape = (uint64_t *)malloc(sizeof(*this->shape) * ndim);
    this->strides = (uint64_t *)malloc(sizeof(*this->strides) * ndim);
    if (this->strides == NULL || this->shape == NULL)
    {
        throw std::runtime_error("Malloc failure in copy.");
    }
    this->shape[0] = vector.size();
    this->shape[1] = row_size;

    this->strides[0] = row_size * this->itemsize;
    this->strides[1] = this->itemsize;
    this->elementCount = vector.size() * row_size;
    setDeviceProperties();
    cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    double *temp = this->buf;
    for (uint64_t i = 0; i < vector.size(); i++)
    {
        err = cudaMemcpy(temp, vector[i].data(), vector[i].size() * this->itemsize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        temp += row_size;
    }
}

DoubleTensor::DoubleTensor(uint64_t emptyAmount)
{
    this->len = emptyAmount * this->itemsize;
    this->ndim = 0;
    this->shape = NULL;
    this->strides = NULL;
    this->elementCount = emptyAmount;
    setDeviceProperties();
    cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

DoubleTensor::~DoubleTensor()
{
    if (this->shape != NULL)
    {
        free(this->shape);
    }
    if (this->strides != NULL)
    {
        free(this->strides);
    }
    cudaError_t err;
    err = cudaSetDevice(this->deviceNumber);
    if (err != cudaSuccess)
    {
        std::cout << "Error setting device to one tensor is on." << std::endl;
    }
    err = cudaFree(this->buf);
    if (err != cudaSuccess)
    {
        std::cout << "Error freeing data from GPU." << std::endl;
    }
}

void DoubleTensor::setIndex(double val, uint64_t index)
{
    if (index >= this->elementCount)
    {
        throw std::runtime_error("Out of Bounds.");
    }
    cudaError_t err = cudaMemcpy((void *)&this->buf[index], &val, sizeof(val), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cout << "Error Setting value on GPU scalar." << std::endl;
    }
}

double DoubleTensor::getIndex(uint64_t index)
{
    if (index >= this->elementCount)
    {
        throw std::runtime_error("Out of Bounds.");
    }
    double val;
    cudaError_t err = cudaMemcpy((void *)&val, (void *)&this->buf[index], sizeof(val), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << "Error Setting value on GPU scalar." << std::endl;
    }

    return val;
}

DoubleTensor *DoubleTensor::deepCopy() const
{
    DoubleTensor *tensor = new DoubleTensor(this->elementCount);
    tensor->len = this->len;
    tensor->ndim = this->ndim;
    tensor->shape = (uint64_t *)malloc(tensor->ndim * sizeof(double));
    tensor->strides = (uint64_t *)malloc(tensor->ndim * sizeof(double));
    if (tensor->strides == NULL || tensor->shape == NULL)
    {
        throw std::runtime_error("Malloc failure in copy.");
    }
    for (uint64_t i = 0; i < tensor->ndim; i++)
    {
        tensor->shape[i] = this->shape[i];
        tensor->strides[i] = this->strides[i];
    }
    tensor->cudaProperties = this->cudaProperties;
    tensor->deviceNumber = this->deviceNumber;
    cudaError_t err = cudaMalloc((void **)&tensor->buf, tensor->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate memory for new array");
    }
    err = cudaMemcpy((void *)tensor->buf, (void *)this->buf, tensor->len, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy over memory in same GPU");
    }
    return tensor;
}

DoubleTensor::DoubleTensor(const std::vector<uint64_t> &vector)
{
    uint64_t elementCount = 1;
    for (auto a : vector)
    {
        elementCount = elementCount * a;
    }
    this->len = elementCount * this->itemsize;
    this->ndim = vector.size();
    this->shape = (uint64_t *)malloc(this->ndim * sizeof(double));
    this->strides = (uint64_t *)malloc(this->ndim * sizeof(double));
    if (this->strides == NULL || this->shape == NULL)
    {
        throw std::runtime_error("Malloc failure in copy.");
    }
    for (uint64_t i = 0; i < this->ndim; i++)
    {
        this->shape[i] = vector[i];
    }
    uint64_t temp;
    for (uint64_t i = 0; i < this->ndim; i++)
    {
        temp = this->itemsize;
        for (uint64_t j = i + 1; j < this->ndim; j++)
        {
            temp = temp * this->shape[j];
        }
        this->strides[i] = temp;
    }
    setDeviceProperties();
    cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate memory for new array");
    }
}

__global__ void CUDA_MEM_EQUAL(const double *ptr1, const double *ptr2, const uint64_t elementCount, int *val)
{
    uint64_t index = blockIdx.x + blockDim.x * blockIdx.x;
    if (index < elementCount)
    {
        if (ptr1[index] != ptr2[index])
        {
            *val = 1;
        }
        printf("Hello \n\n\n\n\n\n");
    }
}

bool DoubleTensor::equals(const DoubleTensor &other) noexcept
{
    try
    {
        if (this->elementCount != other.elementCount)
        {
            return false;
        }
        int *x;
        cudaMalloc((void **)&x, sizeof(int));
        int y = 0;
        cudaMemcpy(x, &y, sizeof(int), cudaMemcpyHostToDevice);
        uint64_t d = this->elementCount / this->cudaProperties.maxThreadsPerBlock <= 1
                         ? 1
                         : this->elementCount / this->cudaProperties.maxThreadsPerBlock;
        CUDA_MEM_EQUAL<<<d, this->cudaProperties.maxThreadsPerBlock>>>(this->buf, other.buf, this->elementCount, x);
        cudaMemcpy(&y, x, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(x);
        return y == 1 ? false : true;
        ;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}

bool DoubleTensor::operator==(const DoubleTensor &other) const noexcept
{
    try
    {
        if (this->elementCount != other.elementCount)
        {
            return false;
        }
        int *x;
        cudaMalloc((void **)&x, sizeof(int));
        int y = 0;
        cudaMemcpy(x, &y, sizeof(int), cudaMemcpyHostToDevice);
        uint64_t d = this->elementCount / this->cudaProperties.maxThreadsPerBlock <= 1
                         ? 1
                         : this->elementCount / this->cudaProperties.maxThreadsPerBlock;
        CUDA_MEM_EQUAL<<<d, this->cudaProperties.maxThreadsPerBlock>>>(this->buf, other.buf, this->elementCount, x);
        cudaMemcpy(&y, x, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(x);
        return y == 1 ? false : true;
        ;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}

std::string DoubleTensor::toString(bool Debug) const
{
    std::ostringstream s;
    if (Debug)
    {
        s << "Data location: " << this->buf << std::endl;
        s << "Length in Bytes: " << this->len << std::endl;
        s << "Element Count: " << this->elementCount << std::endl;
        s << "Number of Dimensions: " << this->ndim << std::endl;
        s << "Shape: (";
        for (int i = 0; i < this->ndim; i++)
        {
            s << this->shape[i];
            if (i < this->ndim - 1)
                s << ", ";
        }
        s << ")" << std::endl;
        s << "Strides: (";
        for (int i = 0; i < this->ndim; i++)
        {
            s << this->strides[i];
            if (i < this->ndim - 1)
                s << ", ";
        }
        s << ")" << std::endl;
        s << "Device: cuda:" << this->deviceNumber << std::endl;
    }
    return s.str();
}