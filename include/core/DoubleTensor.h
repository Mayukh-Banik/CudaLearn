#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nanobind/ndarray.h>

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
    // length in bytes
    uint64_t len;
    const bool readOnly = false;
    const uint64_t itemsize = sizeof(double);
    const std::string format = "d";
    int ndim;
    uint64_t *shape;
    uint64_t *strides;
    // End Python buffer protocol section

    int deviceNumber;
    uint64_t elementCount;

    /**
     * Sets cudaProperties, and deviceNumber.
     */
    void setDeviceProperties();

    DoubleTensor(double value);
    // Init a 1D vector
    DoubleTensor(std::vector<double> vector);
    DoubleTensor(std::vector<std::vector<double>> vector);
    // emulating tuple
    DoubleTensor(const std::vector<uint64_t> &vector);
    // Used for private init, don't bind to python
    DoubleTensor(uint64_t emptyAmount);

    DoubleTensor *T() const;

    DoubleTensor *deepCopy() const;

    ~DoubleTensor();

    void setIndex(double val, uint64_t index);
    double getIndex(uint64_t index);

    bool equals(const DoubleTensor &other) noexcept;
    // bool equals(nanobind::ndarray<double, nanobind::c_contig> other) noexcept;

    bool operator==(const DoubleTensor &other) const noexcept;

    std::string toString(bool Debug = false) const;
};