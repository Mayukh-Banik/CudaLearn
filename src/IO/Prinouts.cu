#include "defs/DoubleTensor.h"
#include <sstream>
#include <cuda_runtime_api.h>
#include <cstdlib>

const std::string DoubleTensor::toString(bool Debug)
{
    std::ostringstream s;
    if (Debug)
    {
        s << "Pointer Address: " << this->Data << std::endl;
        s << "Element Count: " << this->ElementCount << std::endl;
        s << "Shape: (" << this->Shape[0] << ", " << this->Shape[1] << ")" << std::endl;
        s << "Strides: (" << this->Strides[0] << ", " << this->Strides[1] << ")" << std::endl;
        s << "Device: " << (OnGPU ? "cuda" : "cpu") << std::endl;
    }
    if (this->OnGPU)
    {
        double* arr = (double*) calloc(this->Shape[0], this->ItemSize);
        s << "On GPU so printing only first row.";
        cudaError_t err = cudaMemcpy(arr, this->Data, this->Shape[0] * this->ItemSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            s << "Error copying over data from device\n";
            return s.str();
        }
        s << "[\n[";
        for (uint64_t i = 0; i < this->Shape[0]; i++)
        {
            s << arr[i] << ", ";
        }
        s << "]\n]";
        return s.str();
    }
    s << "[\n";
    for (uint64_t i = 0; i < Shape[0]; i++)
    {
        s << "[";
        for (uint64_t j = 0; j < Shape[1]; j++)
        {
            s << Data[i * Shape[1] + j];
            if (j < Shape[1] - 1)
            {
                s << ", ";
            }
        }
        s << "]";
        if (i < Shape[0] - 1)
        {
            s << "\n";
        }
    }
    s << "\n]";
    return s.str();
}