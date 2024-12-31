#include "defs/DoubleTensor.h"

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

/**
 * @brief Tensor from a scalar value, default CPU
 *
 * @details Creates a Tensor from the supplied double value,
 *  internally its always stored as a Matrix. Strides are special to be 0,0
 *  but once exported to a numpy array it will be (8,8)
 */
DoubleTensor::DoubleTensor(double val, std::string Device)
{
    this->Shape = {1, 1};
    this->Strides = {0, 0};
    this->ElementCount = 1;
    this->OnGPU = Device != "cpu";
    this->Device = Device;
    if (OnGPU)
    {
        cudaError_t err = cudaMalloc((void **)&Data, ItemSize);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        err = cudaMemcpy(this->Data, (void *)&val, this->ItemSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    else
    {
        this->Data = (double *)malloc(this->ItemSize);
        if (this->Data == NULL)
        {
            throw std::runtime_error("Memory allocation failed: " + errno);
        }
        this->Data[0] = val;
    }
}

DoubleTensor::DoubleTensor(std::vector<uint64_t> Shape, std::string Device)
{
    switch (Shape.size())
    {
    case 1:
        Shape.push_back(1);
        break;
    case 2:
        break;
    default:
        throw std::invalid_argument("Number of dims > 2.");
    }
    if (Shape[0] == 0 || Shape[1] == 0)
    {
        throw std::invalid_argument("Shape vals must be >0.");
    }
    this->Shape = Shape;
    this->Strides = {this->Shape[1] * this->ItemSize, this->ItemSize};
    this->ElementCount = this->Shape[0] * this->Shape[1];
    this->OnGPU = Device != "cpu";
    this->Device = Device;
    if (OnGPU)
    {
        cudaError_t err = cudaMalloc((void **)&Data, this->ItemSize * this->ElementCount);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    else
    {
        this->Data = (double *)malloc(this->ItemSize * this->ElementCount);
        if (this->Data == NULL)
        {
            throw std::runtime_error("Memory allocation failed: " + errno);
        }
    }
}

DoubleTensor::DoubleTensor(std::vector<std::vector<double>> Value, std::string Device)
{
    if (Value.empty())
    {
        throw std::invalid_argument("The vector is empty.");
    }

    size_t cols = Value[0].size();
    for (const std::vector<double> row : Value)
    {
        if (row.size() != cols)
        {
            throw std::invalid_argument("Inconsistent matrix dimensions: All rows must have the same number of columns.");
        }
    }

    this->Shape = {Value.size(), Value[0].size()};
    this->Strides = {this->Shape[1] * this->ItemSize, this->ItemSize};
    this->ElementCount = this->Shape[0] * this->Shape[1];
    this->OnGPU = Device != "cpu";
    this->Device = Device;
    if (OnGPU)
    {
        cudaError_t err = cudaMalloc((void **)&Data, this->ItemSize * this->ElementCount);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        for (size_t i = 0, index = 0; i < Value.size(); i++, index = index + Value[i].size())
        {
            err = cudaMemcpy(Data + index, Value[i].data(), Value[i].size() * this->ItemSize, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }
    else
    {
        this->Data = (double *)malloc(this->ItemSize * this->ElementCount);
        if (this->Data == NULL)
        {
            throw std::runtime_error("Memory allocation failed: " + errno);
        }
        for (size_t i = 0, index = 0; i < Value.size(); i++, index = index + Value[i].size())
        {
            std::memcpy(this->Data + index, Value[i].data(), Value[i].size() * this->ItemSize);
        }
    }
}

DoubleTensor::~DoubleTensor()
{
    if (OnGPU)
    {
        cudaFree(this->Data);
    }
    else
    {
        free(this->Data);
    }
}

inline void DoubleTensor::getDeviceProperties()
{
    cudaDeviceProp prop;
    int device;
    if (cudaGetDevice(&device) == cudaSuccess)
    {
        cudaGetDeviceProperties(&prop, device);

        this->deviceProperties.MaxGridSize[0] = prop.maxGridSize[0];
        this->deviceProperties.MaxGridSize[1] = prop.maxGridSize[1];
        this->deviceProperties.MaxGridSize[2] = prop.maxGridSize[2];

        this->deviceProperties.MaxThreadsDim[0] = prop.maxThreadsDim[0];
        this->deviceProperties.MaxThreadsDim[1] = prop.maxThreadsDim[1];
        this->deviceProperties.MaxThreadsDim[2] = prop.maxThreadsDim[2];

        this->deviceProperties.MaxThreadsPerBlock = prop.maxThreadsPerBlock;
        this->deviceProperties.WarpSize = prop.warpSize;
    }
}
