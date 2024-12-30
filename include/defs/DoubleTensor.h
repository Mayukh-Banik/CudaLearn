#pragma once

#include <cstdint>
#include <vector>
#include <string>


class DoubleTensor
{
private:
public:
    double *Data;
    uint64_t ElementCount;
    const uint8_t ItemSize = sizeof(*Data);
    std::vector<uint64_t> Shape;
    std::vector<uint64_t> Strides;
    std::string Device;
    bool OnGPU;

    DoubleTensor(double val, std::string Device = "cpu");

    DoubleTensor(std::vector<uint64_t> Shape, std::string Device = "cpu");

    DoubleTensor(std::vector<std::vector<double>> Values, std::string Device = "cpu");

    ~DoubleTensor();
};
