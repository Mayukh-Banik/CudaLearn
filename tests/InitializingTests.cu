#include <gtest/gtest.h>
#include "defs/DoubleTensor.h"
#include <cuda_runtime_api.h>

class DoubleTensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test scalar constructor with default CPU device
TEST_F(DoubleTensorTest, ScalarConstructorDefaultDevice) {
    double value = 42.0;
    DoubleTensor tensor(value);
    
    EXPECT_EQ(tensor.Device, "cpu");
    EXPECT_EQ(tensor.ItemSize, sizeof(double));
    EXPECT_EQ(tensor.Shape.size(), 2);
    EXPECT_EQ(tensor.Shape[0], 1);
    EXPECT_EQ(tensor.Shape[1], 1);
    EXPECT_EQ(tensor.Strides[0], 0);
    EXPECT_EQ(tensor.Strides[1], 0);
    EXPECT_EQ(tensor.ElementCount, 1);
    EXPECT_FALSE(tensor.OnGPU);
    EXPECT_EQ(*tensor.Data, value);
}

// Test scalar constructor with explicit GPU device
TEST_F(DoubleTensorTest, ScalarConstructorGPU) {
    double value = 42.0;
    DoubleTensor tensor(value, "gpu");
    double result;
    
    EXPECT_EQ(tensor.Device, "gpu");
    EXPECT_EQ(tensor.ItemSize, sizeof(double));
    EXPECT_EQ(tensor.Shape.size(), 2);
    EXPECT_EQ(tensor.Shape[0], 1);
    EXPECT_EQ(tensor.Shape[1], 1);
    EXPECT_EQ(tensor.Strides[0], 0);
    EXPECT_EQ(tensor.Strides[1], 0);
    EXPECT_EQ(tensor.ElementCount, 1);
    EXPECT_TRUE(tensor.OnGPU);
    
    cudaError_t err = cudaMemcpy(&result, tensor.Data, sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(result, value);
}

// Test shape constructor with default device
TEST_F(DoubleTensorTest, ShapeConstructorDefaultDevice) {
    std::vector<uint64_t> shape = {2, 3};
    DoubleTensor tensor(shape);
    
    EXPECT_EQ(tensor.Device, "cpu");
    EXPECT_EQ(tensor.ItemSize, sizeof(double));
    EXPECT_EQ(tensor.Shape, shape);
    EXPECT_EQ(tensor.ElementCount, 6);
    EXPECT_EQ(tensor.Strides[0], 3 * sizeof(double));
    EXPECT_EQ(tensor.Strides[1], sizeof(double));
    EXPECT_FALSE(tensor.OnGPU);
}

// Test shape constructor with 1D input
TEST_F(DoubleTensorTest, ShapeConstructor1D) {
    std::vector<uint64_t> shape = {3};
    DoubleTensor tensor(shape);
    
    EXPECT_EQ(tensor.Shape.size(), 2);
    EXPECT_EQ(tensor.Shape[0], 3);
    EXPECT_EQ(tensor.Shape[1], 1);
    EXPECT_EQ(tensor.ElementCount, 3);
    EXPECT_EQ(tensor.Strides[0], sizeof(double));
    EXPECT_EQ(tensor.Strides[1], sizeof(double));
}

// Test shape constructor with invalid dimensions
TEST_F(DoubleTensorTest, ShapeConstructorInvalidDims) {
    std::vector<uint64_t> shape = {2, 3, 4};
    EXPECT_THROW({
        try {
            DoubleTensor tensor(shape);
        } catch(const std::invalid_argument& e) {
            EXPECT_STREQ("Number of dims > 2.", e.what());
            throw;
        }
    }, std::invalid_argument);
}

// Test vector constructor with default device
TEST_F(DoubleTensorTest, VectorConstructorDefaultDevice) {
    std::vector<std::vector<double>> value = {{1.0, 2.0}, {3.0, 4.0}};
    DoubleTensor tensor(value);
    
    EXPECT_EQ(tensor.Device, "cpu");
    EXPECT_EQ(tensor.ItemSize, sizeof(double));
    EXPECT_EQ(tensor.Shape[0], 2);
    EXPECT_EQ(tensor.Shape[1], 2);
    EXPECT_EQ(tensor.ElementCount, 4);
    EXPECT_EQ(tensor.Strides[0], 2 * sizeof(double));
    EXPECT_EQ(tensor.Strides[1], sizeof(double));
    EXPECT_FALSE(tensor.OnGPU);
    
    EXPECT_EQ(tensor.Data[0], 1.0);
    EXPECT_EQ(tensor.Data[1], 2.0);
    EXPECT_EQ(tensor.Data[2], 3.0);
    EXPECT_EQ(tensor.Data[3], 4.0);
}

// Test vector constructor with empty input
TEST_F(DoubleTensorTest, VectorConstructorEmpty) {
    std::vector<std::vector<double>> value;
    EXPECT_THROW({
        try {
            DoubleTensor tensor(value);
        } catch(const std::invalid_argument& e) {
            EXPECT_STREQ("The vector is empty.", e.what());
            throw;
        }
    }, std::invalid_argument);
}

// Test vector constructor with inconsistent dimensions
TEST_F(DoubleTensorTest, VectorConstructorInconsistentDims) {
    std::vector<std::vector<double>> value = {{1.0, 2.0}, {3.0}};
    EXPECT_THROW({
        try {
            DoubleTensor tensor(value);
        } catch(const std::invalid_argument& e) {
            EXPECT_STREQ("Inconsistent matrix dimensions: All rows must have the same number of columns.", e.what());
            throw;
        }
    }, std::invalid_argument);
}

// Test GPU memory handling
TEST_F(DoubleTensorTest, GPUMemoryManagement) {
    // Create a GPU tensor
    std::vector<std::vector<double>> value = {{1.0, 2.0}, {3.0, 4.0}};
    DoubleTensor* tensor = new DoubleTensor(value, "gpu");
    
    EXPECT_TRUE(tensor->OnGPU);
    EXPECT_EQ(tensor->Device, "gpu");
    
    // Verify data is on GPU
    std::vector<double> result(4);
    cudaError_t err = cudaMemcpy(result.data(), tensor->Data, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(result[0], 1.0);
    EXPECT_EQ(result[1], 2.0);
    EXPECT_EQ(result[2], 3.0);
    EXPECT_EQ(result[3], 4.0);
    
    // Test destructor (memory deallocation)
    delete tensor;
    // Note: We can't directly test if memory was freed, but we can verify no crashes
}

// Test ItemSize constant
TEST_F(DoubleTensorTest, ItemSizeConstant) {
    DoubleTensor tensor(1.0);
    EXPECT_EQ(tensor.ItemSize, sizeof(double));
}