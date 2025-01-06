#pragma once

#include <cstdint>
#include <string>

class DoubleTensor
{
private:
public:
    struct
    {
        char name[256];
        uint32_t warpSize;
        uint32_t maxThreadsPerBlock;
        uint32_t maxThreadsDim[3];
        uint32_t maxGridSize[3];
        uint32_t maxGrids;
    } cudaProperties;

    // Python buffer protocol section"
    double *buf;
    uint64_t len;
    int readonly;
    const uint64_t itemsize = sizeof(double);
    const std::string format = "d";
    int ndim;
    uint64_t *shape;
    uint64_t *strides;
    // End Python buffer protocol section

    int deviceNumber;
    uint64_t elementCount;
    bool onGPU;

    /**
     * Sets cudaProperties, and deviceNumber.
     */
    void setDeviceProperties();

    DoubleTensor(double value, int device = 0, int readonly = 0);

    ~DoubleTensor();

    double& operator[](uint64_t index);

    void setIndex(double val, uint64_t index);
    double getIndex(uint64_t index);

};