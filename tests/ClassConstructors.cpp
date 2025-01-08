#include <gtest/gtest.h>

#include "core/DoubleTensor.h"
#include <vector>

TEST(SingleValue, Baisc)
{
    DoubleTensor *a;
    ASSERT_NO_THROW(a = new DoubleTensor(1.0));
    ASSERT_EQ(a->getIndex(0), 1.0);
    ASSERT_NO_THROW(delete (a));
}

TEST(SingleValue, SetInBounds)
{
    DoubleTensor *a;
    ASSERT_NO_THROW(a = new DoubleTensor(1.0));
    ASSERT_NO_THROW(a->setIndex(5.0, 0));
    ASSERT_DOUBLE_EQ(a->getIndex(0), 5.0);
    ASSERT_NO_THROW(delete (a));
}

TEST(SingleValue, SetOutOfBounds)
{
    DoubleTensor *a;
    ASSERT_NO_THROW(a = new DoubleTensor(1.0));
    ASSERT_THROW(a->setIndex(5, 10), std::runtime_error);
    ASSERT_NO_THROW(delete (a));
}

TEST(SingleValue, Properties)
{
    DoubleTensor *a;
    ASSERT_NO_THROW(a = new DoubleTensor(1.0));
    ASSERT_EQ(a->shape, (uint64_t *)NULL);
    ASSERT_EQ(a->strides, (uint64_t *)NULL);
    ASSERT_EQ(a->ndim, 0);
    ASSERT_EQ(a->len, 8);
    ASSERT_EQ(a->elementCount, 1);
    ASSERT_NE(a->buf, (double *)NULL);
    ASSERT_NO_THROW(delete (a));
}

TEST(VectorConstructor, OneDimensionalVector)
{
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0};
    DoubleTensor *a;

    ASSERT_NO_THROW(a = new DoubleTensor(vec));
    ASSERT_EQ(a->elementCount, 4);
    ASSERT_EQ(a->ndim, 1);
    ASSERT_EQ(a->shape[0], 4);

    for (uint64_t i = 0; i < a->elementCount; ++i)
    {
        ASSERT_DOUBLE_EQ(a->getIndex(i), vec[i]);
    }
    ASSERT_NO_THROW(delete (a));
}

TEST(VectorConstructor, TwoDimensionalVector)
{
    std::vector<std::vector<double>> vec = {
        {1.0, 2.0},
        {3.0, 4.0}};
    DoubleTensor *a;

    ASSERT_NO_THROW(a = new DoubleTensor(vec));
    ASSERT_EQ(a->elementCount, 4);
    ASSERT_EQ(a->ndim, 2);
    ASSERT_EQ(a->shape[0], 2);
    ASSERT_EQ(a->shape[1], 2);
    ASSERT_EQ(a->strides[0], 16);
    ASSERT_EQ(a->strides[1], 8);

    ASSERT_DOUBLE_EQ(a->getIndex(0), 1.0);
    ASSERT_DOUBLE_EQ(a->getIndex(1), 2.0);
    ASSERT_DOUBLE_EQ(a->getIndex(2), 3.0);
    ASSERT_DOUBLE_EQ(a->getIndex(3), 4.0);

    ASSERT_NO_THROW(delete (a));
}
