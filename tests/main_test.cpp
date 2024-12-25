#include <gtest/gtest.h>
#include "temp.h"

// Example test case
TEST(SimpleTest, 1plus1) {
    EXPECT_EQ(add(1,1), 2);  // Test that 1 + 1 equals 2
}

// Another test case
TEST(SimpleTest, 3plus3) {
    EXPECT_EQ(add(3,3), 6);  // Test that 2 * 3 equals 6
}
