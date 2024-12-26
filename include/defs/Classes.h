#pragma once

#include <vector>
#include <cstdlib>
#include <cstdint>

template <typename T>
class Tensors
{
private:
public:
    uint64_t Ndim;
    uint64_t* Dimensions;
    uint64_t Elementcount;
    T* Data;

    Tensors(uint64_t Ndim = 0, uint64_t* Dimensions = NULL)
    {
        if (Ndim == 0)
        {
            this->Ndim = 2;
            this->Elementcount = 1;
            this->Dimensions = (uint64_t*) calloc(2, sizeof(int64_t));
            this->Dimensions[0] = 1;
            this->Dimensions[1] = 1;
            this->Data = (T*) calloc(this->Elementcount, sizeof(T));
        }
        else if (Ndim == 1)
        {
            this->Dimensions = (uint64_t*) calloc(sizeof(uint64_t) * 2, 1);                
            this->Dimensions[0] = Dimensions[0];
            this->Elementcount = this->Dimensions[0];
            this->Dimensions[1] = 1;
            this->Data = (T*) calloc(this->Elementcount, sizeof(T));
            this->Ndim = 2;
        }
        else
        {
            this->Ndim = Ndim;
            this->Dimensions = (uint64_t*) calloc(sizeof(uint64_t), Ndim);
            uint64_t ElementCount = 1;
            for (int i = 0; i < Ndim; i++)
            {
                this->Dimensions[i] = Dimensions[i];
                ElementCount = ElementCount * Dimensions[i];
            }
            this->Elementcount = ElementCount;
            this->Data = (T*) calloc(this->Elementcount, sizeof(T));
        }
    }

    ~Tensors()
    {
        free(this->Dimensions);
        free(this->Data);
    }
};