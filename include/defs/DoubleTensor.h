#pragma once

#include <cstdint>
#include <vector>
#include <string>

class DoubleTensor
{
private:
    struct cudaProp
    {
        uint32_t MaxGridSize[3];
        uint32_t MaxThreadsDim[3];
        uint32_t MaxThreadsPerBlock;
        uint32_t WarpSize;
    };
    inline void getDeviceProperties();

public:
    double *Data;
    uint64_t ElementCount;
    const uint8_t ItemSize = sizeof(*Data);
    std::vector<uint64_t> Shape;
    std::vector<uint64_t> Strides;
    std::string Device;
    bool OnGPU;
    cudaProp deviceProperties;
    std::string Order = "C";
    bool ReadOnly = false;

    DoubleTensor(std::string Device);

    DoubleTensor(double val, std::string Device = "cpu");

    DoubleTensor(std::vector<uint64_t> Shape, std::string Device = "cpu");

    DoubleTensor(std::vector<std::vector<double>> Values, std::string Device = "cpu");

    ~DoubleTensor();

    const std::string toString(bool Debug = false);
};
