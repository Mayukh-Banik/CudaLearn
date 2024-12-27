#include "defs/Tensor.h"
#include <algorithm>
#include <cstdlib>

Tensor::Tensor(NumProps props)
{
    this->props = props;
    this->a = malloc(this->props.Size);
    ((uint32*) this->a)[0] = 10;
}

Tensor::~Tensor()
{
    free(this->a);
}

NumProps getSafeType(const NumProps a, const NumProps b)
{
    NumProps c;
    uint8_t val = std::max(a.Size, b.Size);
    if (a.Float || b.Float)
    {
        switch (val)
        {
            case FLOAT.Size:
                c = FLOAT;
                break;
            case DOUBLE.Size:
                c = DOUBLE;
                break;
            case LDOUBLE.Size:
                c = LDOUBLE;
                break;
            default:
                c = DOUBLE;
                break;
        }
        return c;
    }
    if ((a.Signed == false) && (b.Signed == false))
    {
        switch (val)
        {
            case UINT8.Size:
                c = UINT8;
                break;
            case UINT16.Size:
                c = UINT16;
                break;
            case UINT32.Size:
                c = UINT32;
                break;
            case UINT64.Size:
                c = UINT64;
                break;
            default:
                c = UINT32;
                break;
        }
    }
    else
    {
        switch (val)
        {
            case INT8.Size:
                c = INT8;
                break;
            case INT16.Size:
                c = INT16;
                break;
            case INT32.Size:
                c = INT32;
                break;
            case INT64.Size:
                c = INT64;
                break;
            default:
                c = INT32;
                break;
        }
    }
    return c;
}