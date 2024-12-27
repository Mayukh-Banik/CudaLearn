#include "defs/Tensor.h"

void add_int8_from_int8_and_int8(void *a, void *b, void *c)
{
    *((int8 *)c) = *((int8 *)a) + *((int8 *)b);
}

typedef void (*ADD_POINTER)(void *, void *, void *);

struct ADD_FUNCTION
{
    NumProps a;
    NumProps b;
    NumProps c;
    ADD_POINTER pointer;
};

ADD_FUNCTION TABLE[] =
{
    {INT8, INT8, INT8, add_int8_from_int8_and_int8},
};

Tensor *add(Tensor *a, Tensor *b)
{
    Tensor *c = new Tensor(getSafeType(a->props, b->props));
    for (const auto entry : TABLE)
    {
        if (entry.a == a->props && entry.b == b->props && entry.c == c->props)
        {
            entry.pointer(a->a, b->a, c->a);
            break;
        }
    }
    return c;
}
