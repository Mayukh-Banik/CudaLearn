#pragma once

#include <cstdint>

class NumProps
{
private:
public:
    bool Signed;
    bool Float;
    uint8_t Size;

    bool operator==(const NumProps &other) const
    {
        return Signed == other.Signed &&
            Float == other.Float &&
            Size == other.Size;
    }

    bool operator!=(const NumProps &other) const
    {
        return !(*this == other);
    }
};

constexpr NumProps INT8 = {true, false, sizeof(int8_t)};
constexpr NumProps INT16 = {true, false, sizeof(int16_t)};
constexpr NumProps INT32 = {true, false, sizeof(int32_t)};
constexpr NumProps INT64 = {true, false, sizeof(int64_t)};

constexpr NumProps UINT8 = {false, false, sizeof(uint8_t)};
constexpr NumProps UINT16 = {false, false, sizeof(uint16_t)};
constexpr NumProps UINT32 = {false, false, sizeof(uint32_t)};
constexpr NumProps UINT64 = {false, false, sizeof(uint64_t)};

constexpr NumProps FLOAT = {true, true, sizeof(float)};
constexpr NumProps DOUBLE = {true, true, sizeof(double)};
constexpr NumProps LDOUBLE = {true, true, sizeof(long double)};

NumProps getSafeType(const NumProps a, const NumProps b);

class Tensor
{
private:
public:
    NumProps props;
    void *a;

    Tensor(NumProps props = INT32);

    ~Tensor();
};

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using ldouble = long double;
