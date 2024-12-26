#include <gtest/gtest.h>

#include "defs/Classes.h"

TEST(CreateWorks, Init)
{
    EXPECT_ANY_THROW(Tensors<int16_t>());
}