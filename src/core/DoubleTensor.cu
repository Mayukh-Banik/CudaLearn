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
    if (deviceNumber == 0)
    {
        throw std::runtime_error("No CUDA GPUS found");
    }
    err = cudaGetDevice(&deviceNumber);
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

DoubleTensor::DoubleTensor(double val, int device, int readonly)
{
    this->len = this->itemsize;
    this->ndim = 0;
    this->shape = NULL;
    this->strides = NULL;
    this->readonly = readonly;
    this->onGPU = device == 0 ? false : true;
    this->elementCount = 1;
    if (device == 0)
    {
        this->buf = (double *)malloc(this->itemsize);
        if (this->buf == NULL)
        {
            throw new std::runtime_error("Malloc returned an error for allocating sizeof(double).");
        }
    }
    else
    {
        setDeviceProperties();
        cudaError_t err = cudaMalloc((void **)&this->buf, this->itemsize);
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
    if (this->onGPU)
    {
        free(this->buf);
    }
    else
    {
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
}

double& DoubleTensor::operator[](uint64_t index)
{
    return this->buf[index];
}

void DoubleTensor::setIndex(double val, uint64_t index)
{
    if (this->onGPU)
    {
        cudaError_t err = cudaMemcpy((void*) &this->buf[index], &val, sizeof(val), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            std::cout << "Error Setting value on GPU scalar." << std::endl;
        }
    }
    else
    {
        this->buf[index] = val;
    }
}

double DoubleTensor::getIndex(uint64_t index)
{
    double val;
    if (this->onGPU)
    {
        cudaError_t err = cudaMemcpy((void*) &val,(void*) &this->buf[index], sizeof(val), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            std::cout << "Error Setting value on GPU scalar." << std::endl;
        }
    }
    else
    {
        val = this->buf[index];
    }
    return val;
}