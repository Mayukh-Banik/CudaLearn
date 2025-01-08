#include "core/DoubleTensor.h"

#include "cuda_runtime_api.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <iostream>

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
    this->shape[0] = vector.size();
    this->strides = (uint64_t *)malloc(sizeof(*this->shape) * ndim);
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
    this->shape[0] = vector.size();
    this->shape[1] = row_size;
    this->strides = (uint64_t *)malloc(sizeof(*this->strides) * ndim);
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